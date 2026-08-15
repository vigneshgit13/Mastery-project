import logging
from pathlib import Path

from app.ai.pipeline_factory import (
    create_face_pipeline,
)


logging.basicConfig(
    level=logging.INFO
)


def main():

    print("=" * 80)
    print("FULL FACE PIPELINE TEST")
    print("=" * 80)

    # ----------------------------------------------------------
    # Test image
    # ----------------------------------------------------------

    image_path = Path(
        "temp/b0ae1220-5117-4d73-8fae-766031dc8283.jpg"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    print(
        f"Image: {image_path}"
    )

    print()

    # ----------------------------------------------------------
    # Create pipeline
    # ----------------------------------------------------------

    pipeline = create_face_pipeline()

    print()

    # ----------------------------------------------------------
    # Process image
    # ----------------------------------------------------------

    records = pipeline.process(
        image_path
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print()
    print("=" * 80)
    print("PIPELINE RESULT")
    print("=" * 80)

    print(
        f"Total faces: {len(records)}"
    )

    print()

    # ----------------------------------------------------------
    # Individual faces
    # ----------------------------------------------------------

    for record in records:

        print("-" * 80)

        print(
            f"Face {record.face_index:02d}"
        )

        print(
            f"Detection confidence : "
            f"{record.detection_confidence:.4f}"
        )

        print(
            f"BBox                 : "
            f"{record.bbox}"
        )

        print(
            f"Gender               : "
            f"{record.gender.label} "
            f"({record.gender.confidence:.4f})"
        )

        print(
            f"Age                  : "
            f"{record.age.years} years"
        )

        print(
            f"Age group            : "
            f"{record.age.group}"
        )
        if record.age.confidence is not None:
            print(
            f"Age confidence       : "
            f"{record.age.confidence:.4f}"
        )
        else:
            print(
                 "Age confidence       : "
                  "N/A"
            )




        print(
            f"Embedding dimensions : "
            f"{len(record.embedding)}"
        )

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    male_count = sum(
        1
        for record in records
        if record.gender.label.lower() == "male"
    )

    female_count = sum(
        1
        for record in records
        if record.gender.label.lower() == "female"
    )

    child_count = sum(
        1
        for record in records
        if record.age.group.lower() == "child"
    )

    adult_count = sum(
        1
        for record in records
        if record.age.group.lower() == "adult"
    )

    print()
    print("=" * 80)
    print("GENDER COUNTS")
    print("=" * 80)

    print(
        f"Male   : {male_count}"
    )

    print(
        f"Female : {female_count}"
    )

    print()
    print("=" * 80)
    print("AGE GROUP COUNTS")
    print("=" * 80)

    print(
        f"Adult  : {adult_count}"
    )

    print(
        f"Child  : {child_count}"
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    print()
    print("=" * 80)
    print("PIPELINE VALIDATION")
    print("=" * 80)

    embedding_errors = [
        record
        for record in records
        if len(record.embedding) != 512
    ]

    if embedding_errors:

        print(
            f"ERROR: "
            f"{len(embedding_errors)} "
            f"embeddings are not 512-D."
        )

    else:

        print(
            "✓ All embeddings are 512-D"
        )

    invalid_age_groups = [
        record
        for record in records
        if record.age.group not in {
            "Adult",
            "Child",
        }
    ]

    if invalid_age_groups:

        print(
            f"ERROR: "
            f"{len(invalid_age_groups)} "
            f"invalid age groups."
        )

    else:

        print(
            "✓ All age groups are Adult/Child"
        )

    print()
    print("=" * 80)
    print("FULL PIPELINE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()