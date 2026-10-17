from statsbombpy import sb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import warnings

warnings.filterwarnings("ignore")


def load_corner_events():
    """💡 What does this function do?

    This function transforms raw corner events into a structured feature set. For each corner, it extracts the start and end coordinates from StatsBomb's 120x80 yard coordinate system.
    delivery_zone classifies where the ball lands: near post (y < 30), central (30-50), or far post (y > 50).
    delivery_length measures the Euclidean distance from the corner flag to the delivery target.
    is_inswinging and is_outswinging capture the curl direction, which affects how attackers attack the ball.
    is_complete uses StatsBomb's convention where a NaN pass_outcome means the pass was successful.

💡 How does the xG linking work?
    Every event in StatsBomb data belongs to a possession sequence. If a shot occurs in the same possession as the corner, it means the corner delivery led to that shot attempt.
    The function captures the highest shot_statsbomb_xg value from any shot in that possession. Corners that did not produce a shot get an xG of 0. This gives you a danger metric for each delivery pattern."""

    """Load all corner kick events from Euro 2024."""
    matches = sb.matches(competition_id=55, season_id=282)
    print(f"Loaded {len(matches)} matches from UEFA Euro 2024")

    all_corners = []
    all_events_by_match = {}

    for _, match in matches.iterrows():
        match_id = match["match_id"]
        events = sb.events(match_id=match_id)
        all_events_by_match[match_id] = events

        # Filter for corner kick deliveries only
        corners = events[events["pass_type"] == "Corner"].copy()
        corners["match_id"] = match_id
        all_corners.append(corners)

    corners_df = pd.concat(all_corners, ignore_index=True)
    print(f"Extracted {len(corners_df)} corner kicks")
    return corners_df, all_events_by_match


def engineer_features(corners_df, all_events_by_match):
    """Engineer tactical features from corner kick data."""
    features = []

    for _, corner in corners_df.iterrows():
        loc = corner["location"]
        end_loc = corner["pass_end_location"]

        if loc is None or end_loc is None:
            continue
        if not isinstance(loc, list) or not isinstance(end_loc, list):
            continue
        if len(loc) < 2 or len(end_loc) < 2:
            continue

        start_x, start_y = loc[0], loc[1]
        end_x, end_y = end_loc[0], end_loc[1]

        # Euclidean distance from corner flag to delivery target
        delivery_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

        # Classify delivery zone based on where the ball lands
        if end_y < 30:
            delivery_zone = "near_post"
        elif end_y > 50:
            delivery_zone = "far_post"
        else:
            delivery_zone = "central"

        # Delivery type flags
        is_high = 1 if corner.get("pass_height") == "High Pass" else 0
        is_inswinging = 1 if corner.get("pass_inswinging") is True else 0
        is_outswinging = 1 if corner.get("pass_outswinging") is True else 0
        is_cross = 1 if corner.get("pass_cross") is True else 0

        # Pass outcome: NaN means successful in StatsBomb data
        outcome = corner.get("pass_outcome")
        is_complete = 1 if pd.isna(outcome) else 0

        # Link to resulting shots via possession number
        match_id = corner["match_id"]
        possession_num = corner.get("possession")
        match_events = all_events_by_match[match_id]

        xg = 0.0
        led_to_shot = 0
        possession_events = match_events[match_events["possession"] == possession_num]
        shots = possession_events[possession_events["type"] == "Shot"]
        if len(shots) > 0:
            led_to_shot = 1
            xg = shots["shot_statsbomb_xg"].max()
            if pd.isna(xg):
                xg = 0.0

        features.append(
            {
                "match_id": match_id,
                "team": corner.get("team"),
                "player": corner.get("player"),
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "delivery_length": delivery_length,
                "delivery_zone": delivery_zone,
                "is_high": is_high,
                "is_inswinging": is_inswinging,
                "is_outswinging": is_outswinging,
                "is_cross": is_cross,
                "is_complete": is_complete,
                "led_to_shot": led_to_shot,
                "xg": xg,
                "event_id": corner.get("id"),
                "possession": possession_num,
            }
        )

    features_df = pd.DataFrame(features)
    print(f"Engineered features for {len(features_df)} corners")
    print(
        f"  Corners leading to a shot: {features_df['led_to_shot'].sum()} "
        f"({features_df['led_to_shot'].mean() * 100:.1f}%)"
    )
    print(f"  Average xG per corner: {features_df['xg'].mean():.4f}")
    return features_df

def cluster_corners(features_df):
    """💡 What does this code do?
    cluster_features picks the six columns that describe where and how each corner is delivered. Categorical text like delivery_zone is excluded because K-Means needs numeric inputs.
    fillna(0) replaces any missing values with 0 so the model does not crash on NaN entries.
    StandardScaler() transforms each column to have mean 0 and standard deviation 1. Without this, delivery_length (values like 20-40) would overpower is_high (values 0 or 1)."""
    """Cluster corner kicks using K-Means."""
    # Select spatial and tactical features for clustering
    cluster_features = ["end_x", "end_y", "delivery_length", "is_high", "is_inswinging", "is_outswinging"]
    X = features_df[cluster_features].copy()
    X = X.fillna(0)

    # Standardize so no single feature dominates distance calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        # Test different cluster counts to find the elbow
    inertias = []
    k_range = range(3, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Use 4 clusters based on elbow analysis
    optimal_k = 4
    print(f"\nUsing k={optimal_k} clusters")

    # Fit the final model and assign cluster labels
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    features_df["cluster"] = km_final.fit_predict(X_scaled)

    # Print summary statistics per cluster
    print("\nCluster Summary:")
    for c in range(optimal_k):
        cluster_data = features_df[features_df["cluster"] == c]
        print(
            f"  Cluster {c}: {len(cluster_data)} corners, "
            f"shot rate: {cluster_data['led_to_shot'].mean() * 100:.1f}%, "
            f"avg xG: {cluster_data['xg'].mean():.4f}"
        )

    return features_df, km_final, scaler

def visualize_corners(features_df):
    """Generate pitch visualization of corner clusters."""
    pitch = VerticalPitch(half=True, pitch_color="white", line_color="#333333")
    fig, ax = pitch.draw(figsize=(10, 8))

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    cluster_names = []

    for c in sorted(features_df["cluster"].unique()):
        cluster_data = features_df[features_df["cluster"] == c]
        avg_xg = cluster_data["xg"].mean()
        zone = cluster_data["delivery_zone"].mode().iloc[0] if len(cluster_data) > 0 else "unknown"
        cluster_names.append(f"Cluster {c} ({zone}, xG:{avg_xg:.3f})")

        pitch.scatter(
            cluster_data["end_x"],
            cluster_data["end_y"],
            ax=ax,
            color=colors[c % len(colors)],
            s=40,
            alpha=0.5,
            label=cluster_names[-1],
        )

    for c in sorted(features_df["cluster"].unique()):
        cluster_data = features_df[features_df["cluster"] == c]
        centroid_x = cluster_data["end_x"].mean()
        centroid_y = cluster_data["end_y"].mean()
        pitch.scatter(
            [centroid_x],
            [centroid_y],
            ax=ax,
            color=colors[c % len(colors)],
            s=300,
            marker="X",
            edgecolors="black",
            linewidth=1.5,
            zorder=10,
        )

    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("UEFA Euro 2024 - Corner Kick Delivery Clusters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("corner_clusters.png", dpi=150, bbox_inches="tight")
    print("\nSaved: corner_clusters.png")
    plt.close()

def generate_blueprint(features_df):
    """💡 What does this code do?

    For each cluster, the function determines the dominant swing type (inswing, outswing, or mixed) by checking whether more than 50% of corners in that cluster share a trait.
    The blueprint table is sorted by average xG in descending order, so the most dangerous pattern type always appears at rank 1.
    A bar chart is saved as corner_blueprint.png, giving a visual comparison of each pattern's effectiveness."""
    """Generate the Corner Blueprint summary."""
    print("\n" + "=" * 60)
    print("CORNER BLUEPRINT - UEFA Euro 2024")
    print("=" * 60)

    blueprint_data = []
    for c in sorted(features_df["cluster"].unique()):
        cluster_data = features_df[features_df["cluster"] == c]
        count = len(cluster_data)
        shot_rate = cluster_data["led_to_shot"].mean() * 100
        avg_xg = cluster_data["xg"].mean()
        zone = cluster_data["delivery_zone"].mode().iloc[0]

        if cluster_data["is_inswinging"].mean() > 0.5:
            swing = "Inswing"
        elif cluster_data["is_outswinging"].mean() > 0.5:
            swing = "Outswing"
        else:
            swing = "Mixed/Straight"

        height = "Crossed" if cluster_data["is_high"].mean() > 0.5 else "Driven/Short"

        blueprint_data.append(
            {
                "cluster": c,
                "count": count,
                "zone": zone,
                "swing": swing,
                "height": height,
                "shot_rate": shot_rate,
                "avg_xg": avg_xg,
            }
        )

    blueprint_df = pd.DataFrame(blueprint_data)
    blueprint_df = blueprint_df.sort_values("avg_xg", ascending=False)

    print(f"\n{'Rank':<5}{'Type':<30}{'Count':<8}{'Shot%':<8}{'Avg xG':<10}")
    print("-" * 60)
    for rank, (_, row) in enumerate(blueprint_df.iterrows(), 1):
        type_desc = f"{row['swing']} to {row['zone']} ({row['height']})"
        print(f"{rank:<5}{type_desc:<30}{row['count']:<8}{row['shot_rate']:.1f}%{'':>3}{row['avg_xg']:.4f}")

    print("\n" + "=" * 60)

    # Bar chart comparing cluster effectiveness
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = range(len(blueprint_df))
    bars = ax.bar(
        x_pos,
        blueprint_df["avg_xg"],
        color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"][: len(blueprint_df)],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x_pos)
    labels = [
        f"{row['swing']}\n{row['zone']}" for _, row in blueprint_df.iterrows()
    ]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average xG per Corner", fontsize=11)
    ax.set_title("Corner Pattern Effectiveness - UEFA Euro 2024", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("corner_blueprint.png", dpi=150, bbox_inches="tight")
    print("Saved: corner_blueprint.png")
    plt.close()

def compute_defensive_compactness(features_df, all_events_by_match):
    """Compute defensive compactness using 360 freeze-frame data."""
    print("\nLoading 360 freeze-frame data...")

    compactness_values = []
    matches_with_360 = set()

    match_ids = features_df["match_id"].unique()

    for match_id in match_ids:
        try:
            frames = sb.frames(match_id=match_id, fmt="dataframe")
            if frames is not None and len(frames) > 0:
                matches_with_360.add(match_id)

                match_corners = features_df[features_df["match_id"] == match_id]
                for _, corner in match_corners.iterrows():
                    event_id = corner["event_id"]
                    event_frames = frames[frames["id"] == event_id]

                    if len(event_frames) == 0:
                        compactness_values.append(
                            {"event_id": event_id, "compactness": np.nan}
                        )
                        continue

                    defenders = event_frames[event_frames["teammate"].eq(False)]

                    if len(defenders) < 3:
                        compactness_values.append(
                            {"event_id": event_id, "compactness": np.nan}
                        )
                        continue

                    defender_locs = defenders["location"].tolist()
                    points = np.array(
                        [
                            [loc[0], loc[1]]
                            for loc in defender_locs
                            if isinstance(loc, list) and len(loc) >= 2
                        ]
                    )

                    if len(points) < 3:
                        compactness_values.append(
                            {"event_id": event_id, "compactness": np.nan}
                        )
                        continue

                    try:
                        hull = ConvexHull(points)
                        compactness_values.append(
                            {"event_id": event_id, "compactness": hull.volume}
                        )
                    except Exception:
                        compactness_values.append(
                            {"event_id": event_id, "compactness": np.nan}
                        )
        except Exception:
            continue

    print(f"  Matches with 360 data: {len(matches_with_360)}")

    compactness_df = pd.DataFrame(
        compactness_values, columns=["event_id", "compactness"]
    )
    features_df = features_df.merge(compactness_df, on="event_id", how="left")

    valid_compactness = features_df["compactness"].dropna()
    print(f"  Corners with compactness data: {len(valid_compactness)}")

    if len(valid_compactness) > 0:
        print("\n  Average compactness by cluster:")
        for c in sorted(features_df["cluster"].unique()):
            cluster_comp = features_df[
                features_df["cluster"] == c
            ]["compactness"].dropna()
            if len(cluster_comp) > 0:
                print(
                    f"    Cluster {c}: {cluster_comp.mean():.1f} sq units "
                    f"(n={len(cluster_comp)})"
                )

    return features_df

def plot_compactness_vs_danger(features_df):
    """Plot compactness vs xG scatter."""
    plot_data = features_df.dropna(subset=["compactness"])
    plot_data = plot_data[plot_data["xg"] > 0]

    if len(plot_data) < 5:
        print("  Not enough data points with both compactness and xG > 0 for scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    for c in sorted(plot_data["cluster"].unique()):
        cluster_data = plot_data[plot_data["cluster"] == c]
        ax.scatter(
            cluster_data["compactness"],
            cluster_data["xg"],
            color=colors[c % len(colors)],
            s=60,
            alpha=0.7,
            label=f"Cluster {c}",
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Defensive Compactness (Convex Hull Area)", fontsize=11)
    ax.set_ylabel("Expected Goals (xG)", fontsize=11)
    ax.set_title(
        "Defensive Compactness vs Corner Danger - Euro 2024",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("compactness_vs_danger.png", dpi=150, bbox_inches="tight")
    print("  Saved: compactness_vs_danger.png")
    plt.close()


if __name__ == "__main__":
    print("Set-Piece Intelligence Tool")
    print("Analyzing UEFA Euro 2024 Corner Kicks\n")

    corners_df, all_events_by_match = load_corner_events()
    features_df = engineer_features(corners_df, all_events_by_match)
    features_df, model, scaler = cluster_corners(features_df)
    visualize_corners(features_df)
    generate_blueprint(features_df)

    print("\n--- SECRET MISSION: Defensive Compactness Analysis ---")
    features_df = compute_defensive_compactness(features_df, all_events_by_match)
    plot_compactness_vs_danger(features_df)

    print("\n--- Enhanced Blueprint with Defensive Context ---")
    valid = features_df.dropna(subset=["compactness"])
    if len(valid) > 10:
        median_compact = valid["compactness"].median()
        compact_def = valid[valid["compactness"] <= median_compact]
        spread_def = valid[valid["compactness"] > median_compact]
        print(f"  Against COMPACT defense (hull <= {median_compact:.0f}):")
        print(
            f"    Avg xG: {compact_def['xg'].mean():.4f}, "
            f"Shot rate: {compact_def['led_to_shot'].mean() * 100:.1f}%"
        )
        print(f"  Against SPREAD defense (hull > {median_compact:.0f}):")
        print(
            f"    Avg xG: {spread_def['xg'].mean():.4f}, "
            f"Shot rate: {spread_def['led_to_shot'].mean() * 100:.1f}%"
        )

    print("\nDone! All outputs saved.")