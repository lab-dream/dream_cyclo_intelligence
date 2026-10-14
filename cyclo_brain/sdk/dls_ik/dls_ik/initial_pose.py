"""Generate the raised home YAML. This module never publishes robot commands.

Example: python -m dls_ik.initial_pose --urdf robot.urdf --input home.yaml
         --output raised_home.yaml --report initial_pose.json
"""

import argparse
import json
import re
from pathlib import Path

import yaml

from .solver import DLSIK


def read_home(path):
    """Read every ROS wildcard section, including repeated '/**:' keys."""
    text = Path(path).read_text()
    positions = {}
    for section in re.split(r"(?m)(?=^/\*\*:\s*$)", text):
        if not section.strip() or section.lstrip().startswith("#") and "/**:" not in section:
            continue
        document = yaml.safe_load(section)
        if not document or "/**" not in document:
            continue
        for node in document["/**"].values():
            params = node["ros__parameters"]
            for name, value in zip(params["joint_names"], params["home"], strict=True):
                if name in positions:
                    raise ValueError(f"Duplicate initial joint: {name}")
                positions[name] = float(value)
    return positions


def write_home(template, destination, positions):
    """Replace home arrays, retaining ROS sections, names, timings and topics."""
    text = Path(template).read_text()
    sections = re.split(r"(?m)(?=^/\*\*:\s*$)", text)
    for index, section in enumerate(sections):
        if not section.strip() or "/**:" not in section:
            continue
        document = yaml.safe_load(section)
        if not document or "/**" not in document:
            continue
        nodes = document["/**"]
        if len(nodes) != 1:
            raise ValueError("Expected one trajectory executor per ROS wildcard section")
        params = next(iter(nodes.values()))["ros__parameters"]
        values = [round(float(positions[name]), 10) for name in params["joint_names"]]
        section, count = re.subn(r"(?m)^(\s+home:)\s*\[[^\n]*\]", lambda m: m[1] + " " + str(values), section)
        if count != 1:
            raise ValueError("Expected exactly one home array per executor")
        sections[index] = section
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text("".join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    solver = DLSIK(args.urdf)
    seed = read_home(args.input)
    positions, report = solver.initial_pose(seed)
    write_home(args.input, args.output, positions)
    report.update(joint_positions=positions, source=str(args.input), urdf=str(args.urdf))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
