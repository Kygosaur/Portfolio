import wandb
import numpy as np
import plotly.graph_objs as go

def log_custom_line_graph(run, train_marker='circle', val_marker='triangle-up', lr_marker='diamond'):
    """
    Create a custom line graph panel with customizable symbol markers

    :param run: Active Wandb run
    :param train_marker: Marker symbol for train loss (default: 'circle')
    :param val_marker: Marker symbol for validation loss (default: 'triangle-up')
    :param lr_marker: Marker symbol for learning rate (default: 'diamond')
    """
    # Create some example data (replace with your actual metrics)
    epochs = list(range(300))
    train_loss = np.random.randn(300).cumsum() + np.linspace(1, 0, 300)  # Example declining loss
    val_loss = np.random.randn(300).cumsum() + np.linspace(1, 0, 300)
    learning_rate = np.linspace(0.001, 0.0001, 300)

    # Create Plotly traces
    train_trace = go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train Loss', 
                             marker=dict(symbol=train_marker))
    val_trace = go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss', 
                           marker=dict(symbol=val_marker))
    lr_trace = go.Scatter(x=epochs, y=learning_rate, mode='lines+markers', name='Learning Rate', 
                          marker=dict(symbol=lr_marker))

    # Create the layout
    layout = go.Layout(title='Training Metrics', xaxis_title='Epochs', yaxis_title='Values')

    # Create the figure
    fig = go.Figure(data=[train_trace, val_trace, lr_trace], layout=layout)

    # Log the custom plot
    wandb.log({"custom_metrics_panel": wandb.Plotly(fig)})

def train_yolo_model(train_marker='circle', val_marker='triangle-up', lr_marker='diamond'):
    # Initialize wandb run
    run = wandb.init(
        project="yolo-object-detection",
        config={
            "model": "yolov8n",
            "epochs": 300,
            "learning_rate": 0.001
        }
    )

    try:
        # Your existing training code here

        # Log custom line graph with specified markers
        log_custom_line_graph(run, train_marker, val_marker, lr_marker)

    finally:
        # Finish the wandb run
        wandb.finish()

# Run the training
if __name__ == '__main__':
    # You can now specify custom markers when calling train_yolo_model
    train_yolo_model(train_marker='star', val_marker='cross', lr_marker='square')