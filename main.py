from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import uuid
from datetime import datetime
import json
from typing import Dict, List, Optional
import boto3
from pydantic import BaseModel

app = FastAPI(title="🎯 SWSC Shooting Range API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://swsc.nearbe.id", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration - PostgreSQL
DATABASE_URL = os.getenv('DATABASE_URL')  # Railway auto-provides this

def get_db_connection():
    """Create PostgreSQL database connection"""
    try:
        connection = psycopg2.connect(DATABASE_URL)
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def init_database():
    """Initialize database tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create tables
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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR(36) PRIMARY KEY,
                lane_id VARCHAR(36) REFERENCES lanes(id),
                shooter_name VARCHAR(100),
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP NULL,
                total_shots INTEGER DEFAULT 0,
                total_score INTEGER DEFAULT 0,
                status VARCHAR(20) DEFAULT 'active'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shots (
                id VARCHAR(36) PRIMARY KEY,
                session_id VARCHAR(36) REFERENCES sessions(id),
                lane_id VARCHAR(36) REFERENCES lanes(id),
                shot_number INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                coordinates_x FLOAT,
                coordinates_y FLOAT,
                score INTEGER,
                zone VARCHAR(50),
                confidence FLOAT,
                s3_image_url VARCHAR(500),
                target_type VARCHAR(50),
                scoring_system VARCHAR(50)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_lane_status ON sessions(lane_id, status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shots_session_time ON shots(session_id, timestamp)")
        
        # Insert sample data if lanes table is empty
        cursor.execute("SELECT COUNT(*) FROM lanes")
        count = cursor.fetchone()[0]
        
        if count == 0:
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
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        raise

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
        init_database()
        print("🚀 API startup completed")
    except Exception as e:
        print(f"❌ Startup failed: {e}")

@app.get("/")
def read_root():
    return {
        "message": "🎯 SWSC Shooting Range API",
        "version": "1.0.0",
        "status": "online",
        "database": "PostgreSQL",
        "endpoints": [
            "/api/lanes",
            "/api/sessions", 
            "/api/shots",
            "/health"
        ]
    }

@app.get("/health")
def health_check():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "database_type": "PostgreSQL",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# LANES ENDPOINTS
@app.get("/api/lanes")
def get_lanes():
    """Get all active lanes"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT l.*, 
                   COUNT(s.id) as total_sessions,
                   COALESCE(active_s.session_count, 0) as active_sessions
            FROM lanes l
            LEFT JOIN sessions s ON l.id = s.lane_id
            LEFT JOIN (
                SELECT lane_id, COUNT(*) as session_count 
                FROM sessions 
                WHERE status = 'active' 
                GROUP BY lane_id
            ) active_s ON l.id = active_s.lane_id
            WHERE l.active = TRUE
            GROUP BY l.id, active_s.session_count
            ORDER BY l.name
        """)
        
        lanes = cursor.fetchall()
        conn.close()
        
        return {"lanes": [dict(lane) for lane in lanes], "count": len(lanes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lanes")
def create_lane(lane: LaneCreate):
    """Create new lane"""
    try:
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
        
        return {"message": "Lane created successfully", "lane_id": lane_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lanes/{lane_id}")
def get_lane(lane_id: str):
    """Get specific lane details"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM lanes WHERE id = %s", (lane_id,))
        lane = cursor.fetchone()
        
        if not lane:
            raise HTTPException(status_code=404, detail="Lane not found")
        
        conn.close()
        return {"lane": dict(lane)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# SESSIONS ENDPOINTS
@app.get("/api/sessions/{lane_id}")
def get_active_session(lane_id: str):
    """Get active session for lane"""
    try:
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
        
        return {"session": dict(session) if session else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions")
def create_session(session: SessionCreate):
    """Create new shooting session"""
    try:
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
        
        return {"message": "Session created successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/sessions/{session_id}/end")
def end_session(session_id: str):
    """End shooting session"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET status = 'completed', end_time = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (session_id,))
        
        conn.commit()
        conn.close()
        
        return {"message": "Session ended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# SHOTS ENDPOINTS
@app.post("/api/shots")
def record_shot(shot: ShotCreate):
    """Record new shot"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current shot number for session
        cursor.execute("""
            SELECT COALESCE(MAX(shot_number), 0) + 1 as next_shot_number
            FROM shots WHERE session_id = %s
        """, (shot.session_id,))
        shot_number = cursor.fetchone()[0]
        
        # Insert shot
        shot_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO shots (
                id, session_id, lane_id, shot_number, coordinates_x, coordinates_y,
                score, zone, confidence, s3_image_url, target_type, scoring_system
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            shot_id, shot.session_id, shot.lane_id, shot_number,
            shot.coordinates_x, shot.coordinates_y, shot.score, shot.zone,
            shot.confidence, shot.s3_image_url, shot.target_type, shot.scoring_system
        ))
        
        # Update session totals
        cursor.execute("""
            UPDATE sessions 
            SET total_shots = total_shots + 1, total_score = total_score + %s
            WHERE id = %s
        """, (shot.score, shot.session_id))
        
        conn.commit()
        conn.close()
        
        return {"message": "Shot recorded successfully", "shot_id": shot_id, "shot_number": shot_number}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/shots")
def get_session_shots(session_id: str, limit: int = 20):
    """Get shots for a session"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM shots 
            WHERE session_id = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (session_id, limit))
        
        shots = cursor.fetchall()
        conn.close()
        
        return {"shots": [dict(shot) for shot in shots], "count": len(shots)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ANALYTICS ENDPOINTS
@app.get("/api/analytics/dashboard")
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
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
        
        # Recent activity
        cursor.execute("""
            SELECT s.shooter_name, l.name as lane_name, sess.start_time
            FROM sessions sess
            JOIN lanes l ON sess.lane_id = l.id
            LEFT JOIN shots s ON sess.id = s.session_id
            ORDER BY sess.start_time DESC
            LIMIT 5
        """)
        recent_activity = cursor.fetchall()
        
        conn.close()
        
        return {
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
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
