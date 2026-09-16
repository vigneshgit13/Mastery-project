from app.schemas.dashboard import (
    DashboardSummaryResponse,
    DistributionItem,
)


print("=" * 70)
print("WORKFLOW 3 DASHBOARD SCHEMA TEST")
print("=" * 70)


response = DashboardSummaryResponse(
    total_faces=149,
    total_clusters=86,
    total_images=11,
    gender_distribution=[
        DistributionItem(label="Female", count=100),
        DistributionItem(label="Male", count=49),
    ],
    age_group_distribution=[
        DistributionItem(label="Adult", count=145),
        DistributionItem(label="Child", count=4),
    ],
)


print()
print("Schema response:")
print(response.model_dump())

assert response.total_faces == 149
assert response.total_clusters == 86
assert response.total_images == 11

assert sum(
    item.count for item in response.gender_distribution
) == response.total_faces

assert sum(
    item.count for item in response.age_group_distribution
) == response.total_faces

print()
print("Validation:")
print("  Total counts:      PASS")
print("  Gender totals:     PASS")
print("  Age-group totals:  PASS")

print()
print("WORKFLOW 3 DASHBOARD SCHEMA: PASS")