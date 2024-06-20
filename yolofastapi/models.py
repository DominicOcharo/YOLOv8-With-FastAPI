# yolofastapi/models.py

from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ImageAnalysis(Base):
    __tablename__ = "image_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    labels = Column(String, index=True)
    confidences = Column(String)
    filtered_labels = Column(String, nullable=True)
    filtered_confidences = Column(String, nullable=True)
    recommendation = Column(String, nullable=True)
    percentage = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)
