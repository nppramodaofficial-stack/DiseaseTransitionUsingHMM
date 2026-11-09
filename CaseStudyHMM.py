
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("parkinsons_updrs.data")

print("\n===== DATASET INFORMATION =====")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.info()

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== NUMBER OF SUBJECTS =====")
print(df["subject#"].nunique())

print("\n===== OBSERVATIONS PER SUBJECT =====")
print(df["subject#"].value_counts().sort_index())

print("\n===== FIRST FIVE ROWS =====")
print(df.head())



subject_1 = (
    df[df["subject#"] == 1]
    .sort_values("test_time")
)

print("\n===== SUBJECT 1 =====")
print(
    subject_1[
        ["subject#", "test_time", "motor_UPDRS", "total_UPDRS"]
    ].head(20)
)

print("Shape:", subject_1.shape)

if not subject_1.empty:
    print(
        "Test time range:",
        subject_1["test_time"].min(),
        "to",
        subject_1["test_time"].max()
    )

time_order_check = (
    df.groupby("subject#")["test_time"]
    .apply(lambda x: x.is_monotonic_increasing)
)

print("\n===== TIME ORDER CHECK =====")
print(time_order_check)
print("All subjects initially ordered:", time_order_check.all())


data = df[
    ["subject#", "test_time", "motor_UPDRS", "total_UPDRS"]
].copy()

data = data.dropna()

data = data.sort_values(
    ["subject#", "test_time"]
).reset_index(drop=True)

print("\n===== PROCESSED DATA =====")
print(data.head())
print("Shape:", data.shape)

print("\n===== DESCRIPTIVE STATISTICS =====")
print(data[["motor_UPDRS", "total_UPDRS"]].describe())


plt.figure(figsize=(8, 5))
plt.hist(data["motor_UPDRS"], bins=30)
plt.xlabel("Motor UPDRS")
plt.ylabel("Frequency")
plt.title("Distribution of Motor UPDRS")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(data["total_UPDRS"], bins=30)
plt.xlabel("Total UPDRS")
plt.ylabel("Frequency")
plt.title("Distribution of Total UPDRS")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()



subjects = data["subject#"].unique()[:5]

for subject in subjects:
    patient = (
        data[data["subject#"] == subject]
        .sort_values("test_time")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        patient["test_time"],
        patient["motor_UPDRS"],
        marker="o",
        markersize=3
    )
    plt.xlabel("Test Time")
    plt.ylabel("Motor UPDRS")
    plt.title(f"Motor UPDRS Measurements - Subject {subject}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



features = ["motor_UPDRS", "total_UPDRS"]

correlation = data[features].corr()

print("\n===== CORRELATION MATRIX =====")
print(correlation)

plt.figure(figsize=(8, 5))
plt.scatter(
    data["motor_UPDRS"],
    data["total_UPDRS"],
    alpha=0.5
)
plt.xlabel("Motor UPDRS")
plt.ylabel("Total UPDRS")
plt.title("Motor UPDRS vs Total UPDRS")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



scaler = StandardScaler()

scaled_features = scaler.fit_transform(data[features])

data["motor_UPDRS_scaled"] = scaled_features[:, 0]
data["total_UPDRS_scaled"] = scaled_features[:, 1]

print("\n===== STANDARDIZED DATA =====")
print(
    data[
        [
            "subject#",
            "test_time",
            "motor_UPDRS",
            "total_UPDRS",
            "motor_UPDRS_scaled",
            "total_UPDRS_scaled"
        ]
    ].head()
)

print("\nStandardized feature means:")
print(scaled_features.mean(axis=0))

print("\nStandardized feature standard deviations:")
print(scaled_features.std(axis=0))



patient_sequences = {}

for subject, group in data.groupby("subject#"):
    group = group.sort_values("test_time")

    X = group[
        ["motor_UPDRS_scaled", "total_UPDRS_scaled"]
    ].to_numpy()

    patient_sequences[subject] = X

print("\n===== PATIENT OBSERVATION MATRICES =====")
print("Number of subjects:", len(patient_sequences))

first_subject = next(iter(patient_sequences))

print("First subject:", first_subject)
print(
    "Observation matrix shape:",
    patient_sequences[first_subject].shape
)
print("Observation matrix:")
print(patient_sequences[first_subject])


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


data["state_cluster"] = kmeans.fit_predict(scaled_features)

print("\n===== STATE CLUSTER COUNTS =====")
print(data["state_cluster"].value_counts().sort_index())

centers_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)

centers = pd.DataFrame(
    centers_original,
    columns=["motor_UPDRS", "total_UPDRS"]
)

print("\n===== CLUSTER CENTRES IN ORIGINAL UNITS =====")
print(centers)


cluster_severity = (
    data.groupby("state_cluster")[
        ["motor_UPDRS", "total_UPDRS"]
    ].mean()
)

cluster_severity["severity_score"] = (
    cluster_severity["motor_UPDRS"]
    + cluster_severity["total_UPDRS"]
)

severity_order = (
    cluster_severity["severity_score"]
    .sort_values()
    .index
)

state_mapping = {
    old_state: new_state
    for new_state, old_state in enumerate(severity_order)
}

data["hidden_state"] = (
    data["state_cluster"].map(state_mapping).astype(int)
)

print("\n===== STATE SEVERITY SUMMARY =====")
print(cluster_severity)

print("\n===== STATE MAPPING =====")
print(state_mapping)

print("\n===== ORDERED STATE COUNTS =====")
print(data["hidden_state"].value_counts().sort_index())



state_names = {
    0: "Low-score regime",
    1: "Medium-score regime",
    2: "High-score regime"
}

plt.figure(figsize=(8, 5))

for state in range(3):
    subset = data[data["hidden_state"] == state]

    plt.scatter(
        subset["motor_UPDRS"],
        subset["total_UPDRS"],
        alpha=0.45,
        label=state_names[state]
    )

plt.xlabel("Motor UPDRS")
plt.ylabel("Total UPDRS")
plt.title("K-means-Derived Measurement Regimes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



state_sequences = {}

for subject, group in data.groupby("subject#"):
    group = group.sort_values("test_time")
    state_sequences[subject] = group["hidden_state"].to_numpy()

print("\n===== STATE SEQUENCES =====")
print("Number of subject sequences:", len(state_sequences))

first_subject = next(iter(state_sequences))

print("Subject:", first_subject)
print("State sequence:")
print(state_sequences[first_subject])

print("State names:")
print([
    state_names[state]
    for state in state_sequences[first_subject]
])



num_states = 3

transition_counts = np.zeros(
    (num_states, num_states),
    dtype=int
)

for sequence in state_sequences.values():
    for i in range(len(sequence) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]

        transition_counts[current_state, next_state] += 1

print("\n===== TRANSITION COUNT MATRIX =====")
print(transition_counts)


row_sums = transition_counts.sum(axis=1, keepdims=True)

transition_matrix = np.divide(
    transition_counts,
    row_sums,
    out=np.zeros(
        (num_states, num_states),
        dtype=float
    ),
    where=row_sums != 0
)

print("\n===== TRANSITION PROBABILITY MATRIX A =====")
print(transition_matrix)

print("\nRow sums:")
print(transition_matrix.sum(axis=1))

valid_rows = np.isclose(
    transition_matrix.sum(axis=1),
    1.0
)

print("Rows summing to one:", valid_rows)

if not np.all(valid_rows):
    print(
        "WARNING: At least one state has no observed outgoing "
        "transitions. Its transition probabilities are undefined."
    )



state_counts = (
    data["hidden_state"]
    .value_counts()
    .reindex(range(num_states), fill_value=0)
    .sort_index()
)

initial_state_vector = (
    state_counts / state_counts.sum()
).to_numpy(dtype=float)

print("\n===== EMPIRICAL STATE PROBABILITY VECTOR =====")
print(initial_state_vector)
print("Sum:", initial_state_vector.sum())


if np.all(valid_rows):
    next_state_vector = initial_state_vector @ transition_matrix

    print("\n===== NEXT-STEP PROBABILITY VECTOR =====")
    print(next_state_vector)
    print("Sum:", next_state_vector.sum())
else:
    next_state_vector = None
    print(
        "\nOne-step probability vector not calculated because "
        "the transition matrix contains an undefined row."
    )



if np.all(valid_rows):
    n = 10

    transition_matrix_n = np.linalg.matrix_power(
        transition_matrix,
        n
    )

    state_after_n = initial_state_vector @ transition_matrix_n

    print(f"\n===== TRANSITION MATRIX A^{n} =====")
    print(transition_matrix_n)

    print(f"\n===== STATE PROBABILITIES AFTER {n} STEPS =====")
    print(state_after_n)
    print("Sum:", state_after_n.sum())

    state_evolution = [initial_state_vector.copy()]
    current_vector = initial_state_vector.copy()

    for _ in range(n):
        current_vector = current_vector @ transition_matrix
        state_evolution.append(current_vector.copy())

    state_evolution = np.array(state_evolution)

    plt.figure(figsize=(9, 5))

    for state in range(num_states):
        plt.plot(
            range(n + 1),
            state_evolution[:, state],
            marker="o",
            label=state_names[state]
        )

    plt.xlabel("Number of transitions")
    plt.ylabel("State probability")
    plt.title("Evolution of State Probabilities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if np.all(valid_rows):
    eigenvalues, right_eigenvectors = np.linalg.eig(
        transition_matrix
    )

    print("\n===== EIGENVALUES OF A =====")
    print(eigenvalues)

    print("\n===== RIGHT EIGENVECTORS OF A =====")
    print(right_eigenvectors)


    eigenvalues_T, eigenvectors_T = np.linalg.eig(
        transition_matrix.T
    )

    print("\n===== EIGENVALUES OF A TRANSPOSE =====")
    print(eigenvalues_T)

    index = np.argmin(np.abs(eigenvalues_T - 1))

    stationary_vector = np.real(
        eigenvectors_T[:, index]
    )


    if stationary_vector.sum() < 0:
        stationary_vector = -stationary_vector

    if (
        np.all(stationary_vector >= -1e-8)
        and not np.isclose(stationary_vector.sum(), 0)
    ):
        stationary_distribution = (
            stationary_vector / stationary_vector.sum()
        )

        stationary_distribution[
            np.abs(stationary_distribution) < 1e-12
        ] = 0.0

        stationary_distribution = (
            stationary_distribution /
            stationary_distribution.sum()
        )

        print("\n===== STATIONARY DISTRIBUTION =====")
        print(stationary_distribution)
        print("Sum:", stationary_distribution.sum())

        check = stationary_distribution @ transition_matrix

        print("\n===== STATIONARITY VERIFICATION =====")
        print("Original distribution:")
        print(stationary_distribution)

        print("After multiplication by A:")
        print(check)

        print("Difference:")
        print(check - stationary_distribution)

        print(
            "Stationary equation satisfied:",
            np.allclose(
                check,
                stationary_distribution,
                atol=1e-8
            )
        )

    else:
        stationary_distribution = None

        print(
            "\nThe selected eigenvector did not yield a valid "
            "nonnegative stationary probability vector. "
            "Check the transition matrix and its communicating "
            "classes; a unique stationary distribution is not "
            "guaranteed."
        )



if np.all(valid_rows):
    closest_to_one = np.argmin(np.abs(eigenvalues - 1))

    print("\n===== EIGENVALUE CLOSEST TO ONE =====")
    print(eigenvalues[closest_to_one])



print("\n" + "=" * 55)
print("LINEAR ALGEBRA CASE STUDY SUMMARY")
print("=" * 55)

print("Number of subjects:", data["subject#"].nunique())
print("Number of observations:", len(data))
print("Number of states:", num_states)

print("\nTransition count matrix:")
print(transition_counts)

print("\nTransition probability matrix:")
print(transition_matrix)

print("\nEmpirical state probability vector:")
print(initial_state_vector)

if next_state_vector is not None:
    print("\nOne-step probability vector:")
    print(next_state_vector)

if np.all(valid_rows):
    print("\nEigenvalues:")
    print(eigenvalues)

    if stationary_distribution is not None:
        print("\nStationary distribution:")
        print(stationary_distribution)

print("\nAnalysis completed.")
