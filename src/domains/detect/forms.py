from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.fields.simple import SubmitField


class UploadImageForm(FlaskForm):
    image = FileField(
        validators=[
            FileRequired("Image file is required"),
            FileAllowed(
                ["jpg", "png", "jpeg", "gif", "webp", "bmp"],
                "지원하지 않는 파일입니다.",
            ),
        ]
    )
    # model = SelectField(
    #     "모델 선택",
    #     choices=[(de.value, de.name) for de in DetectorEnum],
    #     default=DetectorEnum.FireDetectV1,
    # )
    submit = SubmitField("탐지")


class UploadVideoForm(FlaskForm):
    video = FileField(
        validators=[
            FileRequired("Video file is required"),
            FileAllowed(["mp4", "avi", "mkv"], "지원하지 않는 파일입니다."),
        ]
    )
    # model = SelectField(
    #     "모델 선택",
    #     choices=[(de.value, de.name) for de in DetectorEnum],
    #     default=DetectorEnum.FireDetectV1,
    # )
    submit = SubmitField("Upload")
