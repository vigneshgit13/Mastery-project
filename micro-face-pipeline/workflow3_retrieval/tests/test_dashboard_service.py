from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.dashboard_service import DashboardService


def main() -> None:
    print("=" * 70)
    print("WORKFLOW 3 DASHBOARD SERVICE TEST")
    print("=" * 70)

    db = SessionLocal()

    try:
        repository = DashboardRepository(db)
        service = DashboardService(repository)

        summary = service.get_summary()

        assert summary["total_faces"] == 149
        assert summary["total_clusters"] == 86
        assert summary["total_images"] == 11

        assert (
            sum(summary["gender_distribution"].values())
            == summary["total_faces"]
        )

        assert (
            sum(summary["age_group_distribution"].values())
            == summary["total_faces"]
        )

        print()
        print("Service summary:")
        print(f"  Total faces:     {summary['total_faces']}")
        print(f"  Total clusters:  {summary['total_clusters']}")
        print(f"  Total images:    {summary['total_images']}")

        print()
        print("Validation:")
        print("  Gender totals:    PASS")
        print("  Age-group totals: PASS")

        print()
        print("WORKFLOW 3 DASHBOARD SERVICE: PASS")

    finally:
        db.close()


if __name__ == "__main__":
    main()