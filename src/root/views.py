from flask import Blueprint, render_template

root_views = Blueprint(
    "root",
    __name__,
    template_folder="templates",
    static_folder="static",
)


@root_views.route("/")
def index():
    return render_template("root/index.html")
