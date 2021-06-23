import io
import os
from pydantic import BaseModel,Field,FilePath
from opyrator.components.types import FileContent
from PIL import Image 
imagePath = "/home/shahnaz/Documents/academics/main_project/opyrator/opyrator/"

def loadImage(path,index):
    photo = str(index) + ".jpg"
    path = os.path.join(path,photo)
    
    img = Image.open(path)
    
    img_byte = io.BytesIO()
    img.save(img_byte,format="PNG")
    return img_byte.getvalue()

class ImageNo(BaseModel):
    #image:FileContent = Field(...,mime_type="image/png")
    index:int



class OutputImage(BaseModel):
    image:FileContent = Field(...,mime_type="image/png")
    question:str = Field(
            ...
            )
    answer1: str = Field(
        ...,
        description="Choices for the above question",
        example="He is eating",
        max_length=140,
    )
    answer2: str = Field(
        ...,
        description="Choices for the above question",
        example="He is dancing",
        max_length=140,
    )
    answer3: str = Field(
        ...,
        description="Choices for the above question",
        example="She is sleeping",
        max_length=140,
    )
    answer4: str = Field(
        ...,
        description="Choices for the above question",
        example="The person is cooking",
        max_length=140,
    )

def modelOutput(input:ImageNo)->OutputImage:
    return OutputImage(image=loadImage(imagePath,input.index),question="",answer1="",answer2="",answer3="",answer4="")
