def calculate_metrics(accuracy_matrix):
    """
    Calculate average accuracy, forgetting, and plasticity.

    accuracy_matrix:
        Dictionary mapping task IDs to accuracy values
        measured after each sequential training stage.
    """

    task_count = len(accuracy_matrix)

    # Average accuracy after the final task
    final_accuracies = [
        accuracy_matrix[i][-1]
        for i in range(task_count)
    ]

    average_accuracy = sum(final_accuracies) / task_count

    # Average forgetting across previously learned tasks
    forgetting_values = []

    for i in range(task_count - 1):
        previous_accuracies = accuracy_matrix[i][i:-1]

        if previous_accuracies:
            forgetting = max(previous_accuracies) - accuracy_matrix[i][-1]
            forgetting_values.append(forgetting)

    average_forgetting = (
        sum(forgetting_values) / len(forgetting_values)
        if forgetting_values else 0.0
    )

    # Average accuracy on each task immediately after learning it
    plasticity = sum(
        accuracy_matrix[i][i]
        for i in range(task_count)
    ) / task_count

    return {
        "Average Accuracy": average_accuracy,
        "Average Forgetting": average_forgetting,
        "Plasticity": plasticity
    }