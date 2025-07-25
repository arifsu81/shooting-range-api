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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🎯 SWSC Shooting Range API", version="1.0.0")

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
            time.sleep(1)  # Wait 1 second before retry

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
                total_score INTEGER DEFAULT 0,
                status VARCHAR(20) DEFAULT 'active',
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
                score INTEGER,
                zone VARCHAR(50),
                confidence FLOAT,
                s3_image_url VARCHAR(500),
                target_type VARCHAR(50),
                scoring_system VARCHAR(50),
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
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
    
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

class ShotCreate(BaseModel):
    session_id: str
    lane_id: str
    coordinates_x: float
    coordinates_y: float
    score: int
    zone: str
    confidence: float = 0.0
    s3_image_url: str = ""
    target_type: str = ""
    scoring_system: str = ""

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        logger.info("🚀 API startup initiated...")
        init_database()
        logger.info("🎉 API startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise exception to allow API to start even if DB has issues

@app.get("/")
def read_root():
    return {
        "message": "🎯 SWSC Shooting Range API",
        "version": "1.0.0",
        "status": "online",
        "database": "PostgreSQL",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "lanes": "/api/lanes",
            "sessions": "/api/sessions",
            "shots": "/api/shots",
            "analytics": "/api/analytics/dashboard"
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
        
        health_data = {
            "status": "healthy",
            "database": "connected",
            "database_type": "PostgreSQL",
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
                   COUNT(DISTINCT CASE WHEN s.status = 'active' THEN s.id END) as active_sessions
            FROM lanes l
            LEFT JOIN sessions s ON l.id = s.lane_id
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
            SELECT s.*, l.name as lane_name
            FROM sessions s
            JOIN lanes l ON s.lane_id = l.id
            WHERE s.lane_id = %s AND s.status = 'active'
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
            INSERT INTO sessions (id, lane_id, shooter_name, status)
            VALUES (%s, %s, %s, 'active')
        """, (session_id, session.lane_id, session.shooter_name))
        
        conn.commit()
        conn.close()
        
        result = {
            "message": "Session created successfully", 
            "session_id": session_id,
            "shooter_name": session.shooter_name,
            "lane_id": session.lane_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Session created: {session.shooter_name} ({session_id})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

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
        
        # Recent activity with proper JOIN
        cursor.execute("""
            SELECT sess.shooter_name, l.name as lane_name, sess.start_time
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
                "today_shots": today_shots
            },
            "recent_activity": [
                {
                    "shooter_name": activity[0],
                    "lane_name": activity[1], 
                    "start_time": activity[2].isoformat() if activity[2] else None
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
        
        # Check lane count
        cursor.execute("SELECT COUNT(*) FROM lanes")
        lane_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "database": "connected",
            "tables": tables,
            "lane_count": lane_count,
            "environment": {
                "DATABASE_URL": DATABASE_URL[:50] + "..." if DATABASE_URL else "Not set"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "database": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
