# NFSM Synchronization using Genetic Algorithm

This repository implements a Genetic Algorithm to derive resetting (synchronizing) sequences for Non-deterministic Finite-State Machines (NFSMs). It supports batch processing of NFSMs, detailed result tracking, and customization of evolutionary parameters including chromosome length, population size, and hill climbing optimization.

---

## 🔧 Features

- Evolutionary search for synchronizing sequences in NFSMs
- Configurable genetic algorithm parameters
- Optional hill climbing on top-performing individuals
- Memory and time tracking for performance analysis
- JSON result logging and CLI-based summary report

---

## 🚀 Getting Started

### Prerequisites
- Python 3.6+
- Required files: `SynchWithGA.py`, `Node.py`, and `Nfsm.py`

### Running the Program

```bash
python SynchWithGA.py <input_file> [options]
```

### Example

```bash
python SynchWithGA.py NFSMs.txt -p 100 -i 50 --min_chrom_size 0.6 --max_chrom_size 1.8
```

---

## ⚙️ Command Line Arguments

| Argument                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `input_file`              | **(Required)** Path to the text file containing NFSM definitions            |
| `-o, --output_file`       | Output JSON file to save results (default: `results.json`)                  |
| `-p, --population_size`   | Population size for the GA (default: `60`)                                  |
| `-i, --max_iterations`    | Max generations without improvement before stopping (default: `30`)         |
| `-c, --hill_climb_percent`| Percentage of top individuals to apply hill climbing (default: `0.1`)       |
| `--min_chrom_size`        | Minimum chromosome size multiplier (default: `0.5`)                         |
| `--max_chrom_size`        | Maximum chromosome size multiplier (default: `2.0`)                         |
| `-t, --timeout`           | Timeout in seconds per NFSM (default: `60`)                                 |

---

## 📄 Input File Format

The NFSM input file must have blocks of FSMs structured like this:

```
FSM_ID num_states num_inputs num_outputs saturation num_transitions L
state_1 next_state_1 input_symbol
...
initial_state
```

Each FSM must follow this format and be separated by a newline or additional headers.

---

## 📤 Output

- A JSON file storing:
  - Execution time
  - Memory usage
  - Resetting sequence (if found)
  - FSM attributes and status
- Console summary table of all processed NFSMs

---

## 📜 License

This project is for academic use under the CMP49411 course.

---

## 👨‍💻 Authors

Timothy Joseph
Muhammed Ahmer
