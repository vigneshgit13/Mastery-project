from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository


def main() -> None:
    print("=" * 70)
    print("WORKFLOW 3 DASHBOARD REPOSITORY TEST")
    print("=" * 70)

    db = SessionLocal()

    try:
        repository = DashboardRepository(db)

        summary = repository.get_summary()

        print()
        print("Dashboard summary:")
        print(f"  Total faces:   {summary['total_faces']}")
        print(f"  Total clusters: {summary['total_clusters']}")
        print(f"  Total images:  {summary['total_images']}")

        print()
        print("Gender distribution:")
        for gender, count in summary["gender_distribution"].items():
            print(f"  {gender}: {count}")

        print()
        print("Age-group distribution:")
        for age_group, count in summary["age_group_distribution"].items():
            print(f"  {age_group}: {count}")

        print()
        print("WORKFLOW 3 DASHBOARD REPOSITORY: PASS")

    finally:
        db.close()


if __name__ == "__main__":
    main()