from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy import Column, Integer

Base = declarative_base()

class Base:
    id = Column(Integer, primary_key=True, index=True)
