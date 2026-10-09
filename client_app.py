"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import load_data
from task import test as test_fn

from algorithms import get_trainer

from models import create_model

from utils.fedbn_helper import get_fedbn_state_dict, get_bn_state_dict, load_bn_state_dict
from algorithms.fedpara import convert_to_fedpara

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]
    # Create model
    model = create_model(model_name)
    
    if algorithm == "fedpara":
        convert_to_fedpara(model)

    # Load received parameters
    received_state= msg.content["arrays"].to_torch_state_dict()

    if algorithm == "fedbn":
        # Load shared/global parameters
        model.load_state_dict(received_state, strict=False,)

        # Restore this client's local BatchNorm state
        if "fedbn_bn_state" in context.state:
            print("Restoring local BN state")
            bn_state = context.state["fedbn_bn_state"].to_torch_state_dict()
            load_bn_state_dict(model, bn_state)
        else:
            print("No previous BN state")

    else:
        model.load_state_dict(received_state)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Pull algorithm-specific kwargs (e.g. proximal_mu) straight from config —
    # trainers that don't need them just ignore extras via **kwargs
    algorithm = context.run_config["algorithm"]
    trainer = get_trainer(
        algorithm,
        proximal_mu=msg.content["config"].get("proximal_mu", 0.0),
    )

    # Call the training function
    train_loss = trainer.train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    if algorithm == "fedbn":
        bn_state = get_bn_state_dict(model)
        context.state["fedbn_bn_state"] = ArrayRecord(bn_state)
        print("Saved local BN state")

    # Construct and return reply Message
    if algorithm == "fedbn":
        model_record = ArrayRecord(get_fedbn_state_dict(model))
    else:
        model_record = ArrayRecord(model.state_dict())


    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    
    if hasattr(trainer, "get_metrics"):
        metrics.update(trainer.get_metrics())

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    
    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]

    model = create_model(model_name)
    if algorithm == "fedpara":
        convert_to_fedpara(model)
   
    received_state = msg.content["arrays"].to_torch_state_dict()

    if algorithm == "fedbn":
        model.load_state_dict(received_state, strict=False,)
        if "fedbn_bn_state" in context.state:
            bn_state = context.state["fedbn_bn_state"].to_torch_state_dict()
            load_bn_state_dict(model, bn_state)
    else:
        model.load_state_dict(received_state,)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
