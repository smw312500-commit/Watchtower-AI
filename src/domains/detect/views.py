import cv2
import numpy as np
from flask import (
    Blueprint,
    render_template,
    current_app,
    send_from_directory,
    Response,
)

from src.domains.detect.detector import (
    detector_models,
    DetectorEnum,
)
from src.domains.detect.forms import UploadImageForm

detect_views = Blueprint(
    "detect", __name__, template_folder="templates", static_folder="static"
)


@detect_views.get("/images/<path:filename>")
def images(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@detect_views.route("/images", methods=["GET", "POST"])
def detect_images():
    form = UploadImageForm()
    if form.validate_on_submit():
        file = form.image.data
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        selected_detector = form.model.data
        if selected_detector not in [de.value for de in DetectorEnum]:
            return "잘못된 모델입니다.", 400
        model = detector_models[DetectorEnum(selected_detector)]()
        results = model.detect(img, conf=0.5)

        return ndarray_to_image_bytes(results[0].plot())
    form.model.data = DetectorEnum.FireDetectV1.value
    return render_template("detect/detect_images.html", form=form)


def ndarray_to_image_bytes(array: np.ndarray):
    success, encoded = cv2.imencode(".png", array)
    if not success:
        return "Encode error", 500

    img_bytes = encoded.tobytes()
    return Response(img_bytes, mimetype="image/png")
