

from matplotlib import pyplot as plt
import numpy as np


def chart_labels(labels, total_duration, num_classes):
    """
    Визуализация активности спикеров на временной шкале для Streamlit.
    
    Parameters:
    - labels (array-like): Массив меток кластеров (говорящих).
    - total_duration (float): Общая длительность аудиозаписи в секундах.
    - num_classes (int): Количество классов спикеров.
    """
    labels = np.array(labels)
    num_segments = len(labels)
    time_axis = np.linspace(0, total_duration, num_segments, endpoint=False)

    colors = plt.cm.get_cmap("tab10", num_classes)
    plt.figure(figsize=(12, 6))
    for speaker_id in range(num_classes):
        mask = labels == speaker_id
        plt.scatter(
            time_axis[mask],
            labels[mask],
            label=f"Speaker {speaker_id + 1}",
            color=colors(speaker_id),
            s=100,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Cluster Label")
    plt.title("Speaker Activity Over Time")
    plt.legend(title="Speakers")
    plt.grid()

    # Отображение графика в Streamlit
    return plt
    