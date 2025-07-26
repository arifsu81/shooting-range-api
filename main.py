from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import uuid
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import boto3
from pydantic import BaseModel
import logging
import time
import base64
import io
from PIL import Image
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🎯 SWSC Shooting Range API", version="2.0.0")

# CORS configuration with cache headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://swsc.nearbe.id", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set!")
    raise Exception("DATABASE_URL is required")

# AWS Configuration
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'swsc-shooting-images')

# Initialize AWS clients with error handling
try:
    rekognition = boto3.client('rekognition', region_name=AWS_REGION)
    s3 = boto3.client('s3', region_name=AWS_REGION)
    logger.info("✅ AWS clients initialized successfully")
except Exception as e:
    logger.warning(f"⚠️ AWS initialization warning: {e}")
    rekognition = None
    s3 = None

logger.info(f"Database URL configured: {DATABASE_URL[:50]}...")

def get_db_connection():
    """Create PostgreSQL database connection with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            connection = psycopg2.connect(DATABASE_URL)
            logger.info("✅ Database connection successful")
            return connection
        except Exception as e:
            logger.error(f"❌ Database connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Database connection failed after {max_retries} attempts: {str(e)}")
            time.sleep(1)

def init_database():
    """Initialize database tables with proper error handling"""
    try:
        logger.info("🔄 Initializing database...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create tables with CASCADE for dependencies
        logger.info("Creating lanes table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lanes (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                camera_id VARCHAR(100),
                target_type VARCHAR(50) NOT NULL,
                scoring_system VARCHAR(50) NOT NULL,
                distance_meters INTEGER DEFAULT 25,
                caliber VARCHAR(20),
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.info("Creating sessions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR(36) PRIMARY KEY,
                lane_id VARCHAR(36),
                shooter_name VARCHAR(100),
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP NULL,
                total_shots INTEGER DEFAULT 0,
                total_score FLOAT DEFAULT 0,
                average_score FLOAT DEFAULT 0,
                best_shot FLOAT DEFAULT 0,
                status VARCHAR(20) DEFAULT 'active',
                target_type VARCHAR(50),
                CONSTRAINT fk_sessions_lane FOREIGN KEY (lane_id) REFERENCES lanes(id) ON DELETE CASCADE
            )
        """)
        
        logger.info("Creating shots table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shots (
                id VARCHAR(36) PRIMARY KEY,
                session_id VARCHAR(36),
                lane_id VARCHAR(36),
                shot_number INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                coordinates_x FLOAT,
                coordinates_y FLOAT,
                position_x FLOAT,
                position_y FLOAT,
                score FLOAT NOT NULL DEFAULT 0,
                zone VARCHAR(50),
                confidence FLOAT DEFAULT 0,
                s3_image_url VARCHAR(500),
                target_type VARCHAR(50),
                scoring_system VARCHAR(50),
                holes_detected INTEGER DEFAULT 1,
                analysis_data TEXT,
                CONSTRAINT fk_shots_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                CONSTRAINT fk_shots_lane FOREIGN KEY (lane_id) REFERENCES lanes(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance
        logger.info("Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_lane_status ON sessions(lane_id, status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_session_time ON shots(session_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_lane_time ON shots(lane_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_score ON shots(score)")
        
        # Insert sample data if lanes table is empty
        logger.info("Checking for sample data...")
        cursor.execute("SELECT COUNT(*) FROM lanes")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("Inserting sample lanes...")
            sample_lanes = [
                ('lane-001', 'Lane 1 - Precision', 'camera_1', 'bullseye', 'issf', 25, '.22'),
                ('lane-002', 'Lane 2 - Tactical', 'camera_2', 'silhouette', 'ipsc', 15, '9mm'),
                ('lane-003', 'Lane 3 - Training', 'camera_3', 'bullseye', 'issf', 10, '.177')
            ]
            
            for lane_data in sample_lanes:
                cursor.execute("""
                    INSERT INTO lanes (id, name, camera_id, target_type, scoring_system, distance_meters, caliber)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, lane_data)
            
            logger.info(f"✅ Inserted {len(sample_lanes)} sample lanes")
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

# Cache management middleware
@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Add cache control headers
    if request.url.path.startswith("/api/"):
        # API endpoints - short cache
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    else:
        # Static content - longer cache
        response.headers["Cache-Control"] = "public, max-age=300"
    
    # Add CORS headers for all responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    # Add API timestamp for debugging
    response.headers["X-API-Timestamp"] = datetime.now().isoformat()
    
    return response

# Pydantic models
class LaneCreate(BaseModel):
    name: str
    camera_id: str
    target_type: str
    scoring_system: str
    distance_meters: int = 25
    caliber: str = ""

class SessionCreate(BaseModel):
    lane_id: str
    shooter_name: str
    target_type: str = "bullseye"

class ShotCreate(BaseModel):
    session_id: str
    lane_id: str
    coordinates_x: float = 0
    coordinates_y: float = 0
    score: float
    zone: str = ""
    confidence: float = 0.0
    s3_image_url: str = ""
    target_type: str = ""
    scoring_system: str = ""

class ShotAnalyze(BaseModel):
    session_id: str
    image_data: str  # base64 encoded image
    target_type: str = "bullseye"

class ShotRecord(BaseModel):
    session_id: str
    score: float
    position_x: float
    position_y: float
    confidence: float
    target_type: str
    image_url: Optional[str] = None

# AWS Helper Functions
async def upload_to_s3(image_data: bytes, session_id: str) -> str:
    """Upload image to S3 and return URL"""
    try:
        if not s3:
            logger.warning("S3 client not available")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key = f"shots/session_{session_id}/{timestamp}.jpg"
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=image_data,
            ContentType='image/jpeg'
        )
        
        url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
        logger.info(f"✅ Image uploaded to S3: {key}")
        return url
        
    except Exception as e:
        logger.error(f"❌ S3 upload error: {e}")
        return None

async def analyze_target_image(image_data: bytes, target_type: str) -> dict:
    """Analyze target image using AWS Rekognition (simplified without OpenCV)"""
    try:
        # Try AWS Rekognition first
        if rekognition:
            try:
                response = rekognition.detect_objects(
                    Image={'Bytes': image_data},
                    MaxLabels=10,
                    MinConfidence=60
                )
                logger.info("✅ AWS Rekognition analysis completed")
                
                # Simple analysis based on detected objects
                objects = response.get('ObjectTypes', [])
                confidence = 0.8 if objects else 0.5
                
            except Exception as e:
                logger.warning(f"⚠️ AWS Rekognition failed, using mock analysis: {e}")
                confidence = 0.5
        else:
            confidence = 0.5
        
        # Use mock analysis for scoring (will be replaced with actual CV later)
        score_result = mock_analysis(target_type)
        score_result['confidence'] = confidence
        
        return score_result
        
    except Exception as e:
        logger.error(f"❌ Image analysis error: {e}")
        return mock_analysis(target_type)

def mock_analysis(target_type: str) -> dict:
    """Mock analysis for development (will be replaced with real CV)"""
    
    # Simulate different scoring based on target type
    if target_type == 'bullseye':
        score = random.choice([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        zones = ['X-ring', '9-ring', '8-ring', '7-ring', '6-ring', '5-ring', '4-ring', '3-ring', '2-ring', '1-ring']
        zone = zones[10 - score] if score <= 10 else 'miss'
    elif target_type == 'silhouette':
        score = 5 if random.random() > 0.3 else 0
        zone = 'hit' if score > 0 else 'miss'
    elif target_type == 'hostage':
        choices = [(5, 'target_hit'), (2, 'marginal'), (-5, 'hostage_hit'), (0, 'miss')]
        score, zone = random.choice(choices)
    else:
        score = random.randint(0, 10)
        zone = 'unknown'
    
    # Generate random but realistic position
    if score > 7:  # Good shots closer to center
        x = random.uniform(40, 60)
        y = random.uniform(40, 60)
    elif score > 3:  # Average shots
        x = random.uniform(25, 75)
        y = random.uniform(25, 75)
    else:  # Poor shots
        x = random.uniform(10, 90)
        y = random.uniform(10, 90)
    
    return {
        'score': score,
        'position': {'x': x, 'y': y},
        'confidence': random.uniform(0.6, 0.9),
        'holes_detected': 1,
        'zone': zone,
        'analysis_method': 'mock'  # Indicator this is mock data
    }

def calculate_shooting_analytics(scores: List[float], positions: List[tuple]) -> dict:
    """Calculate comprehensive shooting analytics"""
    try:
        if not scores:
            return {
                "total_shots": 0,
                "average_score": 0,
                "best_shot": 0,
                "accuracy_percentage": 0,
                "group_size": 0,
                "consistency_score": 0,
                "shot_distances": [],
                "horizontal_spread": 0,
                "vertical_spread": 0
            }
        
        total_shots = len(scores)
        average_score = sum(scores) / total_shots
        best_shot = max(scores)
        
        # Calculate accuracy (shots with score > 0)
        hits = len([s for s in scores if s > 0])
        accuracy_percentage = (hits / total_shots) * 100
        
        # Calculate group size (max distance between any two shots)
        group_size = 0
        if len(positions) >= 2:
            max_distance = 0
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    distance = (dx**2 + dy**2)**0.5  # Simple distance without numpy
                    max_distance = max(max_distance, distance)
            group_size = max_distance
        
        # Calculate consistency (inverse of standard deviation)
        if total_shots > 1:
            variance = sum((s - average_score)**2 for s in scores) / total_shots
            std_dev = variance**0.5  # Simple sqrt without numpy
            consistency_score = max(0, 10 - std_dev)
        else:
            consistency_score = 10
        
        # Calculate shot-to-shot distances
        shot_distances = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = (dx**2 + dy**2)**0.5
            shot_distances.append(distance)
        
        # Calculate spreads
        horizontal_spread = 0
        vertical_spread = 0
        if len(positions) >= 2:
            x_values = [p[0] for p in positions]
            y_values = [p[1] for p in positions]
            horizontal_spread = max(x_values) - min(x_values)
            vertical_spread = max(y_values) - min(y_values)
        
        return {
            "total_shots": total_shots,
            "average_score": round(average_score, 2),
            "best_shot": best_shot,
            "accuracy_percentage": round(accuracy_percentage, 1),
            "group_size": round(group_size, 2),
            "consistency_score": round(consistency_score, 2),
            "shot_distances": [round(d, 2) for d in shot_distances],
            "min_distance": round(min(shot_distances), 2) if shot_distances else 0,
            "max_distance": round(max(shot_distances), 2) if shot_distances else 0,
            "avg_distance": round(sum(shot_distances) / len(shot_distances), 2) if shot_distances else 0,
            "horizontal_spread": round(horizontal_spread, 2),
            "vertical_spread": round(vertical_spread, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Analytics calculation error: {e}")
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        logger.info("🚀 API startup initiated...")
        init_database()
        logger.info("🎉 API startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

@app.get("/")
def read_root():
    return {
        "message": "🎯 SWSC Shooting Range API v2.0",
        "version": "2.0.0",
        "status": "online",
        "database": "PostgreSQL",
        "aws_integration": "enabled" if rekognition and s3 else "disabled",
        "computer_vision": "mock_mode",  # Indicates using mock CV
        "timestamp": datetime.now().isoformat(),
        "features": {
            "lanes": "✅ Active",
            "sessions": "✅ Active", 
            "shots": "✅ Active",
            "camera": "✅ Active",
            "ai_scoring": "✅ Mock Mode",
            "analytics": "✅ Active"
        },
        "endpoints": {
            "health": "/health",
            "lanes": "/api/lanes",
            "sessions": "/api/sessions",
            "shots": "/api/shots",
            "analytics": "/api/analytics/dashboard",
            "camera": "/api/shots/analyze"
        }
    }

@app.get("/health")
def health_check():
    try:
        logger.info("🔍 Running health check...")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test, NOW() as server_time")
        result = cursor.fetchone()
        conn.close()
        
        # Check AWS status
        aws_status = "connected" if rekognition and s3 else "disabled"
        
        health_data = {
            "status": "healthy",
            "database": "connected",
            "database_type": "PostgreSQL",
            "aws": aws_status,
            "s3_bucket": S3_BUCKET if s3 else "not_configured",
            "computer_vision": "mock_mode",
            "server_time": result[1].isoformat() if result[1] else None,
            "test_query": "passed",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Health check passed")
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Health check failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# CACHE MANAGEMENT ENDPOINTS
@app.post("/api/cache/clear")
def clear_cache():
    """Clear cache instruction endpoint"""
    return {
        "message": "Cache clear instruction sent",
        "instructions": {
            "browser": "Press Ctrl+F5 or Cmd+Shift+R to hard refresh",
            "api": "Cache-Control headers set to no-cache for API endpoints"
        },
        "headers": {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/cache/status")
def cache_status():
    """Get cache status and headers info"""
    return {
        "cache_policy": {
            "api_endpoints": "no-cache",
            "static_content": "5 minutes",
            "cors": "enabled"
        },
        "headers": {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*"
        },
        "timestamp": datetime.now().isoformat()
    }

# LANES ENDPOINTS
@app.get("/api/lanes")
def get_lanes():
    """Get all active lanes"""
    try:
        logger.info("📋 Fetching lanes...")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT l.*, 
                   COUNT(DISTINCT s.id) as total_sessions,
                   COUNT(DISTINCT CASE WHEN s.status = 'active' THEN s.id END) as active_sessions,
                   COUNT(DISTINCT sh.id) as total_shots
            FROM lanes l
            LEFT JOIN sessions s ON l.id = s.lane_id
            LEFT JOIN shots sh ON l.id = sh.lane_id
            WHERE l.active = TRUE
            GROUP BY l.id, l.name, l.camera_id, l.target_type, l.scoring_system, 
                     l.distance_meters, l.caliber, l.active, l.created_at
            ORDER BY l.name
        """)
        
        lanes = cursor.fetchall()
        conn.close()
        
        result = {
            "lanes": [dict(lane) for lane in lanes], 
            "count": len(lanes),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Retrieved {len(lanes)} lanes")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error fetching lanes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch lanes: {str(e)}")

@app.post("/api/lanes")
def create_lane(lane: LaneCreate):
    """Create new lane"""
    try:
        logger.info(f"🛠️ Creating new lane: {lane.name}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        lane_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO lanes (id, name, camera_id, target_type, scoring_system, distance_meters, caliber)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            lane_id, lane.name, lane.camera_id, lane.target_type,
            lane.scoring_system, lane.distance_meters, lane.caliber
        ))
        
        conn.commit()
        conn.close()
        
        result = {
            "message": "Lane created successfully", 
            "lane_id": lane_id,
            "lane_name": lane.name,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Lane created: {lane.name} ({lane_id})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error creating lane: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create lane: {str(e)}")

@app.get("/api/lanes/{lane_id}")
def get_lane(lane_id: str):
    """Get specific lane details"""
    try:
        logger.info(f"🔍 Fetching lane: {lane_id}")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM lanes WHERE id = %s", (lane_id,))
        lane = cursor.fetchone()
        
        if not lane:
            raise HTTPException(status_code=404, detail="Lane not found")
        
        conn.close()
        
        result = {
            "lane": dict(lane),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Lane retrieved: {lane_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching lane: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch lane: {str(e)}")

# SESSIONS ENDPOINTS
@app.get("/api/sessions/{lane_id}")
def get_active_session(lane_id: str):
    """Get active session for lane"""
    try:
        logger.info(f"🎯 Fetching active session for lane: {lane_id}")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT s.*, l.name as lane_name,
                   COUNT(sh.id) as current_shots,
                   COALESCE(SUM(sh.score), 0) as current_total_score
            FROM sessions s
            JOIN lanes l ON s.lane_id = l.id
            LEFT JOIN shots sh ON s.id = sh.session_id
            WHERE s.lane_id = %s AND s.status = 'active'
            GROUP BY s.id, l.name
            ORDER BY s.start_time DESC 
            LIMIT 1
        """, (lane_id,))
        
        session = cursor.fetchone()
        conn.close()
        
        result = {
            "session": dict(session) if session else None,
            "lane_id": lane_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Session retrieved for lane: {lane_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error fetching session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch session: {str(e)}")

@app.post("/api/sessions")
def create_session(session: SessionCreate):
    """Create new shooting session"""
    try:
        logger.info(f"🎮 Creating session for {session.shooter_name} on lane {session.lane_id}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # End any active session for this lane
        cursor.execute("""
            UPDATE sessions 
            SET status = 'completed', end_time = CURRENT_TIMESTAMP
            WHERE lane_id = %s AND status = 'active'
        """, (session.lane_id,))
        
        # Create new session
        session_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO sessions (id, lane_id, shooter_name, status, target_type)
            VALUES (%s, %s, %s, 'active', %s)
        """, (session_id, session.lane_id, session.shooter_name, session.target_type))
        
        conn.commit()
        conn.close()
        
        result = {
            "message": "Session created successfully", 
            "session_id": session_id,
            "shooter_name": session.shooter_name,
            "lane_id": session.lane_id,
            "target_type": session.target_type,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Session created: {session.shooter_name} ({session_id})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

# SHOTS ENDPOINTS
@app.post("/api/shots")
async def record_shot(shot: ShotRecord):
    """Record a shot with score and position"""
    try:
        logger.info(f"🎯 Recording shot for session {shot.session_id}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current shot number
        cursor.execute("SELECT COUNT(*) FROM shots WHERE session_id = %s", (shot.session_id,))
        shot_number = cursor.fetchone()[0] + 1
        
        # Insert shot record
        shot_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO shots (id, session_id, shot_number, score, position_x, position_y, 
                             confidence, target_type, s3_image_url, zone)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING timestamp
        """, (
            shot_id, shot.session_id, shot_number, shot.score,
            shot.position_x, shot.position_y, shot.confidence,
            shot.target_type, shot.image_url, "unknown"
        ))
        
        timestamp = cursor.fetchone()[0]
        
        # Update session stats
        cursor.execute("""
            UPDATE sessions 
            SET total_shots = total_shots + 1,
                total_score = total_score + %s
            WHERE id = %s
        """, (shot.score, shot.session_id))
        
        conn.commit()
        conn.close()
        
        result = {
            "success": True,
            "shot_id": shot_id,
            "shot_number": shot_number,
            "score": shot.score,
            "timestamp": timestamp.isoformat(),
            "message": "Shot recorded successfully"
        }
        
        logger.info(f"✅ Shot recorded: {shot.score} points")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error recording shot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shots/analyze")
async def analyze_shot_image(shot_data: ShotAnalyze):
    """Analyze shot image using AWS Rekognition and calculate score"""
    try:
        logger.info(f"🔍 Analyzing shot image for session {shot_data.session_id}")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(shot_data.image_data.split(',')[1])
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image data format")
        
        # Upload to S3 for storage
        image_url = await upload_to_s3(image_data, shot_data.session_id)
        
        # Analyze with AWS Rekognition (simplified)
        analysis_result = await analyze_target_image(image_data, shot_data.target_type)
        
        # Record the shot
        shot_record = ShotRecord(
            session_id=shot_data.session_id,
            score=analysis_result['score'],
            position_x=analysis_result['position']['x'],
            position_y=analysis_result['position']['y'],
            confidence=analysis_result['confidence'],
            target_type=shot_data.target_type,
            image_url=image_url
        )
        
        # Save to database
        shot_result = await record_shot(shot_record)
        
        result = {
            "success": True,
            "analysis": analysis_result,
            "shot_record": shot_result,
            "image_url": image_url
        }
        
        logger.info(f"✅ Shot analyzed: {analysis_result['score']} points ({analysis_result.get('analysis_method', 'aws')} method)")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing shot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/shots")
async def get_session_shots(session_id: str):
    """Get all shots for a session"""
    try:
        logger.info(f"📊 Fetching shots for session {session_id}")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT id, shot_number, score, position_x, position_y, confidence, 
                   target_type, timestamp, s3_image_url, zone
            FROM shots 
            WHERE session_id = %s 
            ORDER BY shot_number ASC
        """, (session_id,))
        
        shots = cursor.fetchall()
        conn.close()
        
        shot_list = [dict(shot) for shot in shots]
        
        result = {
            "shots": shot_list,
            "total_shots": len(shot_list),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Retrieved {len(shot_list)} shots for session {session_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error getting session shots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/analysis")
async def get_session_analysis(session_id: str):
    """Get detailed analysis for a session"""
    try:
        logger.info(f"📈 Analyzing session {session_id}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT score, position_x, position_y, timestamp
            FROM shots 
            WHERE session_id = %s 
            ORDER BY timestamp ASC
        """, (session_id,))
        
        shots = cursor.fetchall()
        conn.close()
        
        if not shots:
            return {
                "message": "No shots found for this session",
                "session_id": session_id
            }
        
        # Calculate analytics
        scores = [shot[0] for shot in shots]
        positions = [(shot[1], shot[2]) for shot in shots if shot[1] and shot[2]]
        
        analysis = calculate_shooting_analytics(scores, positions)
        analysis['session_id'] = session_id
        analysis['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"✅ Session analysis completed for {session_id}")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Error getting session analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a shooting session and calculate final stats"""
    try:
        logger.info(f"🏁 Ending session {session_id}")
        
        # Get session analysis
        analysis = await get_session_analysis(session_id)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update session with end time and final stats
        cursor.execute("""
            UPDATE sessions 
            SET end_time = CURRENT_TIMESTAMP,
                total_shots = %s,
                average_score = %s,
                best_shot = %s,
                status = 'completed'
            WHERE id = %s
        """, (
            analysis.get('total_shots', 0),
            analysis.get('average_score', 0),
            analysis.get('best_shot', 0),
            session_id
        ))
        
        conn.commit()
        conn.close()
        
        result = {
            "success": True,
            "message": "Session ended successfully",
            "session_id": session_id,
            "final_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Session ended: {session_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ANALYTICS ENDPOINTS
@app.get("/api/analytics/dashboard")
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        logger.info("📊 Fetching dashboard statistics...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total lanes
        cursor.execute("SELECT COUNT(*) FROM lanes WHERE active = TRUE")
        total_lanes = cursor.fetchone()[0]
        
        # Active sessions
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
        active_sessions = cursor.fetchone()[0]
        
        # Today's shots
        cursor.execute("""
            SELECT COUNT(*) FROM shots 
            WHERE DATE(timestamp) = CURRENT_DATE
        """)
        today_shots = cursor.fetchone()[0]
        
        # Total shots
        cursor.execute("SELECT COUNT(*) FROM shots")
        total_shots = cursor.fetchone()[0]
        
        # Average score today
        cursor.execute("""
            SELECT COALESCE(AVG(score), 0) FROM shots 
            WHERE DATE(timestamp) = CURRENT_DATE
        """)
        avg_score_today = round(cursor.fetchone()[0], 2)
        
        # Recent activity with proper JOIN
        cursor.execute("""
            SELECT sess.shooter_name, l.name as lane_name, sess.start_time,
                   sess.total_shots, sess.total_score
            FROM sessions sess
            JOIN lanes l ON sess.lane_id = l.id
            WHERE sess.shooter_name IS NOT NULL
            ORDER BY sess.start_time DESC
            LIMIT 5
        """)
        recent_activity = cursor.fetchall()
        
        conn.close()
        
        result = {
            "stats": {
                "total_lanes": total_lanes,
                "active_sessions": active_sessions,
                "today_shots": today_shots,
                "total_shots": total_shots,
                "avg_score_today": avg_score_today
            },
            "recent_activity": [
                {
                    "shooter_name": activity[0],
                    "lane_name": activity[1], 
                    "start_time": activity[2].isoformat() if activity[2] else None,
                    "total_shots": activity[3] or 0,
                    "total_score": float(activity[4] or 0)
                }
                for activity in recent_activity
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Dashboard stats retrieved")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error fetching dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard stats: {str(e)}")

@app.get("/api/analytics/live/{session_id}")
async def get_live_analytics(session_id: str):
    """Get real-time analytics for active session"""
    try:
        logger.info(f"📊 Getting live analytics for session {session_id}")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get session info
        cursor.execute("""
            SELECT s.*, l.name as lane_name, l.target_type as lane_target_type
            FROM sessions s
            JOIN lanes l ON s.lane_id = l.id 
            WHERE s.id = %s
        """, (session_id,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        conn.close()
        
        # Get shots analysis
        analysis = await get_session_analysis(session_id)
        
        # Add session info
        analysis['session_info'] = {
            'session_id': session_id,
            'lane_id': session['lane_id'],
            'lane_name': session['lane_name'],
            'shooter_name': session['shooter_name'],
            'start_time': session['start_time'].isoformat() if session['start_time'] else None,
            'target_type': session['target_type'] or session['lane_target_type'],
            'status': session['status']
        }
        
        logger.info(f"✅ Live analytics retrieved for session {session_id}")
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting live analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/shots/today")
def get_today_shots():
    """Get today's shot statistics"""
    try:
        logger.info("📊 Fetching today's shot statistics...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as total_shots,
                   COALESCE(AVG(score), 0) as avg_score,
                   COALESCE(MAX(score), 0) as best_shot,
                   COUNT(DISTINCT session_id) as sessions_today
            FROM shots 
            WHERE DATE(timestamp) = CURRENT_DATE
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "total_shots": result[0],
            "avg_score": round(result[1], 2),
            "best_shot": result[2],
            "sessions_today": result[3],
            "date": datetime.now().date().isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching today's shots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# DEBUG ENDPOINTS
@app.get("/api/debug/database")
def debug_database():
    """Debug database connection and tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Check counts
        cursor.execute("SELECT COUNT(*) FROM lanes")
        lane_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM shots")
        shot_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "database": "connected",
            "tables": tables,
            "counts": {
                "lanes": lane_count,
                "sessions": session_count,
                "shots": shot_count
            },
            "environment": {
                "DATABASE_URL": DATABASE_URL[:50] + "..." if DATABASE_URL else "Not set",
                "AWS_REGION": AWS_REGION,
                "S3_BUCKET": S3_BUCKET
            },
            "aws_status": {
                "rekognition": "available" if rekognition else "not configured",
                "s3": "available" if s3 else "not configured"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "database": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/debug/aws")
def debug_aws():
    """Debug AWS configuration"""
    try:
        aws_info = {
            "region": AWS_REGION,
            "s3_bucket": S3_BUCKET,
            "rekognition_available": rekognition is not None,
            "s3_available": s3 is not None,
            "environment_vars": {
                "AWS_ACCESS_KEY_ID": "configured" if os.getenv('AWS_ACCESS_KEY_ID') else "missing",
                "AWS_SECRET_ACCESS_KEY": "configured" if os.getenv('AWS_SECRET_ACCESS_KEY') else "missing",
                "AWS_DEFAULT_REGION": AWS_REGION,
                "S3_BUCKET_NAME": S3_BUCKET
            }
        }
        
        # Test S3 connection if available
        if s3:
            try:
                s3.head_bucket(Bucket=S3_BUCKET)
                aws_info["s3_bucket_accessible"] = True
            except Exception as e:
                aws_info["s3_bucket_accessible"] = False
                aws_info["s3_error"] = str(e)
        
        # Test Rekognition if available
        if rekognition:
            try:
                rekognition.list_collections(MaxResults=1)
                aws_info["rekognition_accessible"] = True
            except Exception as e:
                aws_info["rekognition_accessible"] = False
                aws_info["rekognition_error"] = str(e)
        
        return aws_info
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
