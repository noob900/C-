# C- Project Portfolio

A collection of robotics, algorithms, utilities, data, image-processing, slicing, and mathematics projects organized by topic.

## Project Structure

```text
C-/
|-- algorithms/            # Problem solving, C++ practice, puzzle scripts
|-- Controls/              # Control-system experiments
|-- data/                  # Small datasets and raw data
|-- hardware_control/      # Camera and hardware interface experiments
|-- image_processing/      # Computer vision and image analysis
|-- mathematics/           # MATLAB/Octave computations
|-- robotics/              # Robot simulations, controls, and URDF models
|-- Slicing/               # Mesh slicing, CNC/toolpath, and CAD workflows
|-- utilities/             # General-purpose utility scripts
|-- outputs/               # Local generated files, ignored by Git
`-- Soletair power PLC test code
```

## Quick Navigation

| Folder | Purpose | Key files |
| --- | --- | --- |
| `robotics/` | Robot simulation, navigation, and controls | `quadbot.py`, `Autonomous.py`, `RRTbase.py` |
| `Slicing/` | Mesh visualization and toolpath generation | `meshvisual.py`, `CNCtoolpath.py` |
| `Slicing/StepRobotFrames/` | CAD-to-robot-frame tooling | `step_to_robot_frames.py`, `cad_processor.py` |
| `algorithms/` | Algorithm practice and puzzle solving | `Mazegeneration.py`, `Puzzlesolve.py` |
| `utilities/` | Plotting, scraping, calculators, ASCII media | `scrapper.py`, `class_plot.py` |
| `hardware_control/` | Hardware and webcam access | `camera acess.py` |
| `image_processing/` | Computer vision filters and processing | `Image_processing.py` |
| `data/` | Datasets and data examples | `economy_of_finland.csv` |
| `mathematics/` | Mathematical modeling | `Applied_mathematics_6_2.m` |

## Getting Started

### Installation

```bash
git clone <your-repo-url>
cd C-
python -m venv .venv
.\.venv\Scripts\activate
pip install pybullet numpy opencv-python matplotlib pandas pygame
```

For the CAD/toolpath workflow:

```bash
pip install -r Slicing/StepRobotFrames/requirements.txt
```

### Run Examples

Quadbot simulation:

```bash
cd robotics
python quadbot.py
```

Maze generation:

```bash
cd algorithms
python Mazegeneration.py
```

Image processing:

```bash
cd image_processing
python Image_processing.py
```

## Project Details

- **Robotics:** PyBullet simulations, URDF models, autonomous navigation, and robot control.
- **Slicing:** Mesh visualization, slicing, CNC/toolpath experiments, and CAD-to-robot-frame utilities.
- **Algorithms:** Competitive-programming practice, maze generation, and puzzle solving.
- **Utilities:** Data processing, plotting, web scraping, and small helper tools.
- **Hardware control:** Camera and device interface experiments.
- **Image processing:** Filters, edge detection, and computer-vision workflows.
- **Data:** Small datasets and analysis examples.
- **Mathematics:** Numerical methods and MATLAB/Octave modeling.

## Tech Stack

- Python 3.x
- C++
- MATLAB/Octave
- HTML/CSS
- PyBullet, OpenCV, NumPy, Pandas, Matplotlib, PyVista, Trimesh, CadQuery

## Repository Notes

- Generated files belong in `outputs/` and are ignored by Git.
- Virtual environments belong in `.venv/` and are ignored by Git.
- Rebuild C++ executables locally instead of committing `.exe` files.
- Python bytecode caches are ignored.

## Contributing

1. Create or update the relevant project folder.
2. Add a short `README.md` when a folder needs setup or usage notes.
3. Keep generated outputs out of commits unless they are intentional deliverables.
4. Test scripts locally before committing.

## License

This project portfolio is organized for learning and development purposes.

**Last updated:** June 25, 2026  
**Status:** Active development
