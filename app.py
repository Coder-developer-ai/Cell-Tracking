"""
Streamlit UI for Cell_Dynamics.

This file is a THIN orchestration layer only. It does not implement any
segmentation / detection / tracking / training logic itself. All processing
is performed by the functions defined in the original Cell_Dynamics.ipynb
notebook, which is loaded and executed in-memory below (no pipeline .py file
is ever created).

    Streamlit UI  -->  Cell_Dynamics.ipynb functions  -->  Cell_Dynamics outputs
"""

import io
import json
import sys
import traceback
import types
import uuid
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import streamlit as st

NOTEBOOK_PATH = Path(__file__).parent / "Cell_Dynamics.ipynb"
UPLOAD_ROOT = Path(__file__).parent / "uploaded_dataset"



def _extract_zip_dataset(uploaded_zip) -> Path:
    target = UPLOAD_ROOT / f"zip_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip) as zf:
        zf.extractall(target)


    entries = list(target.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return target


def _safe_relative_path(name: str) -> Path:
    """Return a safe relative path supplied by a directory upload."""
    normalized = str(name).replace("\\\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError("Invalid uploaded file path.")
    return Path(*parts)


def _save_directory_upload(files, frame_pattern) -> Path:
    """
    Save a real browser-selected folder without requiring ZIP.

    The uploaded relative paths are preserved, then each directory containing
    valid frame PNGs is staged as a Cell_Dynamics sequence directory. This is
    necessary because the notebook's discover_sequences() expects sequence
    folders with tNNNN.png directly inside them.
    """
    upload_root = UPLOAD_ROOT / f"directory_{uuid.uuid4().hex[:8]}"
    source_root = upload_root / "_original"
    staged_root = upload_root / "_staged"
    source_root.mkdir(parents=True, exist_ok=True)
    staged_root.mkdir(parents=True, exist_ok=True)

    # First preserve the complete original folder tree.
    uploaded_paths = []
    for uploaded_file in files:
        relative = _safe_relative_path(uploaded_file.name)
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(uploaded_file.getbuffer())
        uploaded_paths.append(relative)

    # Find every directory (including the root) that directly contains
    # Cell_Dynamics frame files. Images in deeper subfolders become their own
    # sequences. This supports:
    #   folder/t0000.png
    #   folder/subfolder/t0000.png
    #   folder/subfolder/subsubfolder/t0000.png
    # and also images directly in the uploaded folder.
    frame_files_by_dir = {}
    for relative in uploaded_paths:
        if relative.suffix.lower() != ".png":
            continue
        if not frame_pattern.match(relative.name):
            continue
        parent = relative.parent
        frame_files_by_dir.setdefault(parent, []).append(relative)

    if not frame_files_by_dir:
        raise ValueError(
            "No valid Cell_Dynamics PNG frames were found. "
            "Frame names must match the notebook pattern such as t0000.png."
        )

    used_sequence_names = set()
    sequence_map = {}

    for relative_dir, relative_files in sorted(frame_files_by_dir.items(), key=lambda item: str(item[0])):
        # Make a unique, readable sequence ID from the original relative folder.
        raw_name = "root" if str(relative_dir) == "." else str(relative_dir)
        sequence_id = raw_name.replace("\\\\", "/").replace("/", "__")
        sequence_id = "".join(
            ch if ch.isalnum() or ch in "._-" else "_"
            for ch in sequence_id
        ).strip("._-") or "sequence"

        base_id = sequence_id
        counter = 2
        while sequence_id in used_sequence_names:
            sequence_id = f"{base_id}_{counter}"
            counter += 1
        used_sequence_names.add(sequence_id)

        staged_sequence_dir = staged_root / sequence_id
        staged_sequence_dir.mkdir(parents=True, exist_ok=True)

        for relative in sorted(relative_files, key=str):
            source_file = source_root / relative
            # The engine expects the frame itself directly inside the sequence.
            destination = staged_sequence_dir / relative.name
            destination.write_bytes(source_file.read_bytes())

        sequence_map[sequence_id] = str(relative_dir)

    # Keep metadata so the UI can explain how uploaded folders were mapped.
    (upload_root / "sequence_map.json").write_text(
        json.dumps(sequence_map, indent=2),
        encoding="utf-8",
    )

    return staged_root


def _make_output_zip(output_dir: Path) -> bytes:
    """Package the complete output directory, preserving its folder tree."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(output_dir))
    buffer.seek(0)
    return buffer.getvalue()



@st.cache_resource(show_spinner="Loading Cell_Dynamics engine...")

def load_engine():
    try:
        import nbformat
        from pathlib import Path

        notebook_path = Path("Cell_Dynamics.ipynb")

        with notebook_path.open("r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        source = "\n\n".join(
            cell["source"]
            for cell in notebook.cells
            if cell.cell_type == "code"
        )

        module = {}
        exec(compile(source, str(notebook_path), "exec"), module)

        return module

    except ModuleNotFoundError as e:
        st.error(
            f"Missing Python package: {e.name}. "
            "Add it to requirements.txt and redeploy."
        )
        st.stop()

    except Exception as e:
        st.error(f"Engine loading failed: {e}")
        st.stop()

engine = load_engine()

st.set_page_config(page_title="Cell Dynamics", layout="wide")
st.title("🔬 Cell Dynamics")
st.caption("Streamlit front-end for the existing Cell_Dynamics.ipynb pipeline (single processing engine).")
st.info(
    "📁 Upload a real folder without ZIP, including nested subfolders. "
    "📦 After processing, download the complete output containing masks, "
    "tracks, visualizations, and other generated files."
)


st.header("Configuration")

st.subheader("Dataset")
dataset_source = st.radio(
    "Dataset source",
    ["Existing folder path", "Upload images"],
    horizontal=True,
)

dataset_dir = None
if dataset_source == "Existing folder path":
    dataset_dir = st.text_input("Dataset directory", value="dataset-smaller")
else:
    upload_kind = st.radio(
        "Upload type",
        [
            "Real folder / directory (no ZIP)",
            "Zip of sequence folder(s)",
            "PNG frames for a single sequence",
        ],
        horizontal=True,
    )

    if upload_kind == "Real folder / directory (no ZIP)":
        st.caption(
            "Select a real folder directly. PNG files may be in the folder itself, "
            "a subfolder, or a sub-subfolder. No ZIP is required."
        )
        uploaded_directory = st.file_uploader(
            "📁 Upload a real cell-image folder",
            type=["png"],
            accept_multiple_files="directory",
            help=(
                "Choose the folder containing your cell images. "
                "The relative folder structure is preserved."
            ),
        )

        if uploaded_directory:
            upload_signature = "::".join(
                f"{f.name}:{f.size}" for f in uploaded_directory
            )
            cache_key = f"directory::{upload_signature}"

            if st.session_state.get("upload_cache_key") != cache_key:
                try:
                    dataset_dir = _save_directory_upload(
                        uploaded_directory,
                        engine.FRAME_RE,
                    )
                    st.session_state["upload_cache_key"] = cache_key
                    st.session_state["uploaded_dataset_dir"] = str(dataset_dir)
                except Exception as exc:
                    st.session_state["uploaded_dataset_dir"] = None
                    st.error(f"Could not save uploaded folder: {exc}")

            if st.session_state.get("uploaded_dataset_dir"):
                dataset_dir = Path(st.session_state["uploaded_dataset_dir"])
                st.success(
                    f"Uploaded {len(uploaded_directory)} PNG file(s). "
                    "Nested folders are supported."
                )

                original_root = dataset_dir.parent / "_original"
                if original_root.exists():
                    original_files = sorted(
                        p.relative_to(original_root)
                        for p in original_root.rglob("*")
                        if p.is_file()
                    )
                    with st.expander("📂 Uploaded folder structure"):
                        st.code(
                            "\n".join(str(p) for p in original_files[:500]),
                            language="text",
                        )
                        if len(original_files) > 500:
                            st.caption(
                                f"Showing first 500 of {len(original_files)} files."
                            )

    elif upload_kind == "Zip of sequence folder(s)":
        st.caption(
            "Zip should contain one folder per sequence, each holding that "
            "sequence's PNG frames."
        )
        uploaded_zip = st.file_uploader("Upload dataset .zip", type=["zip"])
        if uploaded_zip is not None:
            cache_key = f"zip::{uploaded_zip.name}::{uploaded_zip.size}"
            if st.session_state.get("upload_cache_key") != cache_key:
                dataset_dir = _extract_zip_dataset(uploaded_zip)
                st.session_state["upload_cache_key"] = cache_key
                st.session_state["uploaded_dataset_dir"] = str(dataset_dir)
            dataset_dir = Path(st.session_state["uploaded_dataset_dir"])
            st.success(f"Extracted to `{dataset_dir}`")

    else:
        seq_name = st.text_input("Sequence folder name", value="uploaded_sequence")
        uploaded_frames = st.file_uploader(
            "Upload PNG frames (e.g. t0000.png, t0001.png, ...)",
            type=["png"],
            accept_multiple_files=True,
        )
        if uploaded_frames:
            cache_key = (
                f"frames::{seq_name}::{len(uploaded_frames)}::"
                f"{sum(f.size for f in uploaded_frames)}"
            )
            if st.session_state.get("upload_cache_key") != cache_key:
                dataset_dir = _save_frame_uploads(seq_name, uploaded_frames)
                st.session_state["upload_cache_key"] = cache_key
                st.session_state["uploaded_dataset_dir"] = str(dataset_dir)
            dataset_dir = Path(st.session_state["uploaded_dataset_dir"])
            st.success(
                f"Saved {len(uploaded_frames)} frame(s) to "
                f"`{dataset_dir / seq_name}`"
            )

c1, c2, c3 = st.columns(3)
with c1:
    output_dir = st.text_input("Output directory", value="outputs")
    mode = st.selectbox("Mode", ["infer", "train", "train-infer"], index=0)
with c2:
    segmentation_method = st.selectbox("Segmentation method", ["auto", "model", "classical"], index=0)
    checkpoint_path = st.text_input("Checkpoint path (optional)", value="")
    only_sequence = st.text_input("Sequence (blank = all)", value="")
with c3:
    limit_frames = st.number_input("Frame limit (0 = no limit)", min_value=0, value=0, step=1)
    confidence_threshold = st.slider("Confidence / threshold", 0.0, 1.0, 0.5, 0.01)
    visualize_every = st.number_input("Visualization frequency (every N frames)", min_value=0, value=25, step=1)

if st.button("🔍 Discover sequences in dataset directory", disabled=dataset_dir is None):
    try:
        probe_cfg = engine.Config(dataset_dir=Path(dataset_dir), limit_frames=None, only_sequence=None)
        found = engine.discover_sequences(probe_cfg)
        st.session_state["discovered_sequences"] = [s.sequence_id for s in found]
    except Exception as exc:
        st.error(f"Could not discover sequences: {exc}")

if st.session_state.get("discovered_sequences"):
    st.write("Found sequences:", ", ".join(st.session_state["discovered_sequences"]))

c4, c5 = st.columns(2)
with c4:
    save_masks = st.checkbox("Save masks", value=True)
with c5:
    save_visualizations = st.checkbox("Save visualizations", value=True)

with st.expander("Training settings (used when mode is 'train' or 'train-infer')"):
    t1, t2, t3 = st.columns(3)
    with t1:
        train_mask_dir = st.text_input("Training mask directory (optional)", value="")
        image_size = st.number_input("Image size", min_value=32, value=512, step=32)
        batch_size = st.number_input("Batch size", min_value=1, value=16, step=1)
    with t2:
        epochs = st.number_input("Epochs", min_value=1, value=50, step=1)
        learning_rate = st.number_input("Learning rate", min_value=0.0, value=1e-3, step=1e-4, format="%.6f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, value=1e-4, step=1e-5, format="%.6f")
    with t3:
        val_split = st.slider("Validation split", 0.0, 0.9, 0.2, 0.05)
        num_workers = st.number_input("Number of workers", min_value=0, value=8, step=1)

run = st.button("▶️ Run Cell Dynamics", type="primary", disabled=dataset_dir is None)
if dataset_dir is None:
    st.info("Provide a dataset directory path or upload images above to enable Run.")

# --------------------------------------------------------------------------
# Orchestration: build Config, then call ONLY existing Cell_Dynamics
# functions in the same sequence the notebook's own main() uses.
# --------------------------------------------------------------------------
if run:
    config = engine.Config(
        dataset_dir=Path(dataset_dir),
        output_dir=Path(output_dir),
        train_mask_dir=Path(train_mask_dir) if train_mask_dir else None,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        image_size=int(image_size),
        batch_size=int(batch_size),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        val_split=float(val_split),
        num_workers=int(num_workers),
        confidence_threshold=float(confidence_threshold),
        segmentation_method=segmentation_method,
        limit_frames=int(limit_frames) if limit_frames else None,
        only_sequence=only_sequence or None,
        visualize_every=int(visualize_every),
        save_masks=save_masks,
        save_visualizations=save_visualizations,
    )

    log_buffer = io.StringIO()
    result = {"mode": mode, "config": config}

    try:
        with st.spinner("Running Cell_Dynamics..."):
            with redirect_stdout(log_buffer):
                engine.set_seed(config.seed)
                engine.create_output_folders(config)

                trained_checkpoint = None
                if mode in {"train", "train-infer"}:
                    trained_checkpoint = engine.train_model(config)
                    result["trained_checkpoint"] = str(trained_checkpoint)
                    if mode == "train":
                        config.checkpoint_path = trained_checkpoint
                    else:
                        config.checkpoint_path = trained_checkpoint

                if mode in {"infer", "train-infer"}:
                    sequences = engine.discover_sequences(config)
                    if not sequences:
                        raise RuntimeError(f"No PNG sequences found under {config.dataset_dir}")

                    model, device = None, None
                    if config.segmentation_method in {"auto", "model"}:
                        checkpoint_candidate = config.checkpoint_path or (config.checkpoint_dir / "best_unet.pt")
                        if checkpoint_candidate.exists():
                            model, device = engine.load_checkpoint_model(config)
                        elif config.segmentation_method == "model":
                            raise FileNotFoundError(
                                f"'model' segmentation requires a checkpoint, none found at {checkpoint_candidate}"
                            )

                    all_frames = [
                        engine.run_sequence(sequence, config, model=model, device=device)
                        for sequence in sequences
                    ]
                    all_tracks = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
                    all_tracks.to_csv(config.tracks_dir / "all_tracks.csv", index=False)
                    engine.save_summary(sequences, all_tracks, config)

                    result["sequences"] = sequences
                    result["all_tracks"] = all_tracks

        result["log"] = log_buffer.getvalue()
        st.session_state["result"] = result
        st.success("Cell_Dynamics run finished.")
    except Exception:
        st.session_state["result"] = None
        st.error("Cell_Dynamics run failed.")
        st.code(log_buffer.getvalue() + "\n" + traceback.format_exc())

result = st.session_state.get("result")
if result:
    st.header("Results")
    config = result["config"]

    if "trained_checkpoint" in result:
        st.subheader("Training")
        st.write(f"Best checkpoint saved to: `{result['trained_checkpoint']}`")
        history_path = config.logs_dir / "training_history.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            st.dataframe(history_df, use_container_width=True)
            if {"dice", "val_loss"}.issubset(history_df.columns):
                st.line_chart(history_df.set_index("epoch")[["dice", "iou", "val_loss", "train_loss"]])

    if "all_tracks" in result:
        sequences = result["sequences"]
        all_tracks = result["all_tracks"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Sequences processed", len(sequences))
        m2.metric("Detections", len(all_tracks))
        m3.metric("Tracks", int(all_tracks["track_id"].nunique()) if not all_tracks.empty else 0)

        summary_path = config.logs_dir / "run_summary.csv"
        if summary_path.exists():
            st.subheader("Run summary (per sequence)")
            st.dataframe(pd.read_csv(summary_path), use_container_width=True)

        st.subheader("All tracks")
        st.dataframe(all_tracks, use_container_width=True)

        all_tracks_csv = config.tracks_dir / "all_tracks.csv"
        if all_tracks_csv.exists():
            st.download_button(
                "⬇️ Download all_tracks.csv",
                data=all_tracks_csv.read_bytes(),
                file_name="all_tracks.csv",
                mime="text/csv",
            )

        st.subheader("Output files")
        output_files = sorted(p for p in Path(config.output_dir).rglob("*") if p.is_file())
        st.write(f"{len(output_files)} file(s) under `{config.output_dir}`")
        st.dataframe(
            pd.DataFrame(
                {
                    "path": [
                        str(p.relative_to(Path(config.output_dir)))
                        for p in output_files
                    ]
                }
            ),
            use_container_width=True,
            height=200,
        )

        if output_files:
            complete_output_zip = _make_output_zip(Path(config.output_dir))
            st.download_button(
                "📦 Download COMPLETE output — masks + tracks + visualizations + logs",
                data=complete_output_zip,
                file_name="cell_dynamics_complete_output.zip",
                mime="application/zip",
                help=(
                    "Downloads the entire output directory while preserving "
                    "its folder structure."
                ),
            )

    with st.expander("Run log"):
        st.code(result.get("log", ""))