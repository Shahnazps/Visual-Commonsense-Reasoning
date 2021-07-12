from typing import List, Dict
import io
from PIL import Image
from pydantic import BaseModel, Field
import numpy as np
import streamlit as st
from opyrator.components.types import FileContent


def load_image(path):
    img = Image.open(path)
    img_byte = io.BytesIO()
    img.save(img_byte, format="PNG")
    return img_byte.getvalue()


some_question: str = "This is some sample question"
answers: List[str] = [f"This is answer {x}" for x in range(1, 5)]


class ImageNo(BaseModel):
    index: int = 0
    # image: FileContent = Field(..., mime_type="image/png")
    question: str = Field(
        ...,
        description="Choices for the above question",
        example=some_question,
        max_length=140,
    )
    answer1: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[0],
        max_length=140,
    )
    answer2: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[1],
        max_length=140,
    )
    answer3: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[2],
        max_length=140,
    )
    answer4: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[3],
        max_length=140,
    )
    user_Answer_Choice: int
    rationale1: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[0],
        max_length=140,
    )
    rationale2: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[1],
        max_length=140,
    )
    rationale3: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[2],
        max_length=140,
    )
    rationale4: str = Field(
        ...,
        description="Choices for the above question",
        example=answers[3],
        max_length=140,
    )
    user_Rationale_Choice: int


class OutputImage(BaseModel):
    #image: FileContent = Field(..., mime_type="image/png")
    label: str
    prob: str
    acc: str


def pre_commands():
    image_path = "bleh.png"
    img = Image.open(image_path)
    st.image(np.array(img))


def post_commands():
    image_path = "bleh.png"
    img = Image.open(image_path)
    st.image(np.array(img))


def modelOutput(input: ImageNo) -> OutputImage:
    image_path = "bleh.png"
    label = "some_label"
    prob = 0.1
    acc = 0.2
    # img = Image.open(image_path)
    # st.image(np.array(img))
    return OutputImage(label=str(label), prob=str(prob), acc=str(acc))
    #return OutputImage(image=load_image(image_path), label=str(label), prob=str(prob), acc=str(acc))
