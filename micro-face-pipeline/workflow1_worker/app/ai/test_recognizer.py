import logging
from pathlib import Path

import cv2
import numpy as np

from app.ai.recognizer import recognizer


logging.basicConfig(
    level=logging.INFO
)


ALIGNED_DIR = Path(
    "temp/aligned"
)


def main():

    print("=" * 80)
    print("ARCFACE EMBEDDING TEST")
    print("=" * 80)

    face_paths = sorted(
        ALIGNED_DIR.glob("face_*.jpg")
    )

    if not face_paths:
        raise RuntimeError(
            f"No aligned faces found in {ALIGNED_DIR}"
        )

    print(
        f"Found {len(face_paths)} aligned faces."
    )

    embeddings = []

    for index, face_path in enumerate(
        face_paths,
        start=1,
    ):

        print()
        print(
            f"Processing face {index}: "
            f"{face_path}"
        )

        face = cv2.imread(
            str(face_path)
        )

        if face is None:
            raise RuntimeError(
                f"Could not read {face_path}"
            )

        print(
            f"Face shape: {face.shape}"
        )

        embedding = recognizer.get_embedding(
            face
        )

        embeddings.append(
            embedding
        )

        print(
            f"Embedding shape: "
            f"{embedding.shape}"
        )

        print(
            f"Embedding dtype: "
            f"{embedding.dtype}"
        )

        print(
            f"Embedding norm: "
            f"{np.linalg.norm(embedding):.6f}"
        )

        print(
            "First 10 values:"
        )

        print(
            embedding[:10]
        )

    # ------------------------------------------------------------
    # Stack embeddings
    # ------------------------------------------------------------

    matrix = np.stack(
        embeddings,
        axis=0,
    )

    print()
    print("=" * 80)
    print("FINAL EMBEDDING MATRIX")
    print("=" * 80)

    print(
        f"Shape: {matrix.shape}"
    )

    print(
        f"Dtype: {matrix.dtype}"
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    print(
        f"Minimum norm: "
        f"{norms.min():.6f}"
    )

    print(
        f"Maximum norm: "
        f"{norms.max():.6f}"
    )

    print()
    print(
        "ARCFACE TEST COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()