"""Pydantic response models for the detection API."""

from pydantic import BaseModel, Field


class BoxModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionModel(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoxModel


class ImageModel(BaseModel):
    width: int
    height: int


class DetectResponse(BaseModel):
    image: ImageModel
    model: str
    device: str
    inference_ms: float
    detections: list[DetectionModel]


class InfoResponse(BaseModel):
    model: str
    backend: str
    device: str
    imgsz: int
    conf: float
    max_detections: int
    ultralytics_version: str
