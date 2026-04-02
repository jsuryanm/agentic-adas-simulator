import os
from src.tools.detection_tool import DetectionTool

def test_detection():
    tool = DetectionTool()
    BASE_DIR = os.path.dirname(__file__)
    img_path = os.path.join(BASE_DIR,"images","test.jpg")
    detected,dims = tool.detect(img_path)

    assert isinstance(detected,list)

    if detected:
        assert detected[0].label is not None 
        assert 0 <= detected[0].confidence <= 1
    
    assert len(dims) == 2
    