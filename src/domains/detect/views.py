import uuid
from pathlib import Path
from typing import List

from flask import (
    Blueprint,
    render_template,
    current_app,
    send_from_directory,
    redirect,
    url_for,
)
from flask_login import current_user, login_required

from src.domains.detect.forms import UploadImageForm, UploadVideoForm
from src.domains.detect.models import UserImage, UserVideo
from src.main import db

detect_views = Blueprint(
    "detect", __name__, template_folder="templates", static_folder="static"
)


@detect_views.get("/images")
def image_dashboard():
    user_images: List[UserImage] = UserImage.query.all()

    return render_template("detect/image_dashboard.html", user_images=user_images)


@detect_views.get("/images/<path:filename>")
def images(filename: str):
    return send_from_directory(
        Path(current_app.config["UPLOAD_FOLDER"], "images"), filename
    )


@detect_views.route("/upload/images", methods=["GET", "POST"])
@login_required
def upload_image():
    form = UploadImageForm()

    if form.validate_on_submit():
        file = form.image.data

        ext = Path(file.filename).suffix
        image_uuid_file_name = str(uuid.uuid4()) + ext
        image_path = Path(
            current_app.config["UPLOAD_FOLDER"], "images", image_uuid_file_name
        )
        file.save(image_path)

        user_image = UserImage(user_id=current_user.id, image_path=image_uuid_file_name)
        db.session.add(user_image)
        db.session.commit()

        return redirect(url_for("detect.image_dashboard"))
    return render_template("detect/upload_images.html", form=form)


@detect_views.route("/images/detect/<int:image_id>")
def detect_images(image_id: int):
    user_image: UserImage = db.get_or_404(UserImage, image_id)

    return render_template("detect/image_detail.html", user_image=user_image)


@detect_views.get("/videos")
def video_dashboard():
    user_videos: List[UserVideo] = UserVideo.query.all()

    return render_template("detect/video_dashboard.html", user_videos=user_videos)


@detect_views.route("/upload/videos", methods=["GET", "POST"])
@login_required
def upload_video():
    form = UploadVideoForm()

    if form.validate_on_submit():
        file = form.video.data

        ext = Path(file.filename).suffix
        video_uuid_file_name = str(uuid.uuid4()) + ext
        video_path = Path(
            current_app.config["UPLOAD_FOLDER"], "videos", video_uuid_file_name
        )
        file.save(video_path)

        user_video = UserVideo(user_id=current_user.id, video_path=video_uuid_file_name)
        db.session.add(user_video)
        db.session.commit()

        return redirect(url_for("detect.video_dashboard"))
    return render_template("detect/upload_videos.html", form=form)


# @detect_views.route("/images", methods=["GET", "POST"])
# def detect_images():
#     form = UploadImageForm()
#     if form.validate_on_submit():
#         file = form.image.data
#         file_bytes = np.frombuffer(file.read(), np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#
#         selected_detector = form.model.data
#         if selected_detector not in [de.value for de in DetectorEnum]:
#             return "잘못된 모델입니다.", 400
#         model = detector_models[DetectorEnum(selected_detector)]()
#         results = model.detect(img, conf=0.5)
#
#         return ndarray_to_image_bytes(results[0].plot())
#     form.model.data = DetectorEnum.FireDetectV1.value
#     return render_template("detect/detect_images.html", form=form)
#
#
# @detect_views.route("/videos", methods=["GET", "POST"])
# def detect_videos():
#     form = UploadVideoForm()
#     if form.validate_on_submit():
#         file = form.video.data
#
#         with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
#             tmp.write(file.read())
#             tmp_path = tmp.name
#
#         cap = cv2.VideoCapture(tmp_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#         fourcc = cv2.VideoWriter.fourcc(*"avc1")
#         out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
#
#         selected_detector = form.model.data
#         if selected_detector not in [de.value for de in DetectorEnum]:
#             return "잘못된 모델입니다.", 400
#         model = detector_models[DetectorEnum(selected_detector)]()
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             results = model.detect(frame, conf=0.5, verbose=False)
#
#             out.write(results[0].plot())
#
#         cap.release()
#         out.release()
#
#         return Response(generate("output.mp4"), mimetype="video/mp4")
#
#     form.model.data = DetectorEnum.FireDetectV1.value
#     return render_template("detect/detect_videos.html", form=form)
#
#
# def generate(filename: str):
#     out_buffer = io.BytesIO()
#
#     with open(filename, "rb") as f:
#         out_buffer.write(f.read())
#
#     out_buffer.seek(0)
#     while True:
#         data = out_buffer.read(1024 * 1024)
#         if not data:
#             break
#         yield data
#
#
# def ndarray_to_image_bytes(array: np.ndarray):
#     success, encoded = cv2.imencode(".png", array)
#     if not success:
#         return "Encode error", 500
#
#     img_bytes = encoded.tobytes()
#     return Response(img_bytes, mimetype="image/png")
