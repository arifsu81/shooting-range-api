from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
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

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'sql.freedb.tech'),
    'user': os.getenv('DB_USER', 'u251802700_userswsc'),
    'password': os.getenv('DB_PASSWORD', '5VWA;]b4^J'),
    'database': os.getenv('DB_NAME', 'u251802700_swsc'),
    'port': int(os.getenv('DB_PORT', '3306'))
}

# AWS Configuration
AWS_CONFIG = {
    'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
    'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'region': os.getenv('AWS_REGION', 'us-east-1'),
    'bucket': os.getenv('S3_BUCKET_NAME')
}

def get_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

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

@app.get("/")
def read_root():
    return {
        "message": "🎯 SWSC Shooting Range API",
        "version": "1.0.0",
        "status": "online",
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
        cursor = conn.cursor(dictionary=True)
        
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
            GROUP BY l.id
            ORDER BY l.name
        """)
        
        lanes = cursor.fetchall()
        conn.close()
        
        return {"lanes": lanes, "count": len(lanes)}
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
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM lanes WHERE id = %s", (lane_id,))
        lane = cursor.fetchone()
        
        if not lane:
            raise HTTPException(status_code=404, detail="Lane not found")
        
        conn.close()
        return {"lane": lane}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# SESSIONS ENDPOINTS
@app.get("/api/sessions/{lane_id}")
def get_active_session(lane_id: str):
    """Get active session for lane"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
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
        
        return {"session": session}
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
            SET status = 'completed', end_time = NOW()
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
            SET status = 'completed', end_time = NOW()
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
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT * FROM shots 
            WHERE session_id = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (session_id, limit))
        
        shots = cursor.fetchall()
        conn.close()
        
        return {"shots": shots, "count": len(shots)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ANALYTICS ENDPOINTS
@app.get("/api/analytics/dashboard")
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Total lanes
        cursor.execute("SELECT COUNT(*) as count FROM lanes WHERE active = TRUE")
        total_lanes = cursor.fetchone()['count']
        
        # Active sessions
        cursor.execute("SELECT COUNT(*) as count FROM sessions WHERE status = 'active'")
        active_sessions = cursor.fetchone()['count']
        
        # Today's shots
        cursor.execute("""
            SELECT COUNT(*) as count FROM shots 
            WHERE DATE(timestamp) = CURDATE()
        """)
        today_shots = cursor.fetchone()['count']
        
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
            "recent_activity": recent_activity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
