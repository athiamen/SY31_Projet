from setuptools import setup, find_packages
from glob import glob

package_name = "sy31_detection"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch",  glob("launch/*.py")),
        (f"share/{package_name}/config",  glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Équipe SY31",
    description="Détection d'objets TurtleBot3 – SY31 Projet P26",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detection_node    = sy31_detection.detection_node:main",
            "transformer       = sy31_detection.transformer:main",
            "intensity_filter  = sy31_detection.intensity_filter:main",
            "clusterer         = sy31_detection.clusterer:main",
            "shaper_bbox       = sy31_detection.shaper_bbox:main",
            "shaper_cylinder   = sy31_detection.shaper_cylinder:main",
            "shaper_polyline   = sy31_detection.shaper_polyline:main",
            "detect            = sy31_detection.detect:main",
        ],
    },
)
