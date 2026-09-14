from app.db.models import Face, FaceEmbedding, Image, ProcessingJob, Upload
from app.db.postgres import check_database_connection

check_database_connection()

print("PostgreSQL connection: OK")
print("ORM models: OK")
print("Upload columns:", [c.name for c in Upload.__table__.columns])
print("Image columns:", [c.name for c in Image.__table__.columns])
print("Face columns:", [c.name for c in Face.__table__.columns])
print("FaceEmbedding columns:", [c.name for c in FaceEmbedding.__table__.columns])
print("ProcessingJob columns:", [c.name for c in ProcessingJob.__table__.columns])
