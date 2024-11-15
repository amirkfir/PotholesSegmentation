import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize import *
from generate_object_proposals import get_batch_selective_search_regions, evaluate_batch_object_proposals, prepare_proposals_images, prepare_proposals_database, generate_and_save_proposals
from object_data_loader import load_and_transform_objects
from torchsummary import summary
from model import Pothole_RCNN
import torch.nn as nn
import torchvision.models as models
import torch

def main():
    ##general parameters
    data_path = '../data/Potholes/'
    image_resize = 128
    batch_size = 50
    IOU_th = 0.7

    # load data
    #trainset, valset, testset, train_loader, val_loader, test_loader = load_and_transform_dataset(val_size=0.05,
    #                                                                                              batch_size=batch_size,
    #                                                                                              image_resize=image_resize,
    #                                                                                              data_path=data_path)

    # test loader
    #images, (objects, num_objects) = next(iter(train_loader))
    #visualize_boxes(images, objects, num_objects)

    #batch_rects = get_batch_selective_search_regions(images)
    #visualize_proposals(images, batch_rects, num_proposals=50)
    #prepare_proposals_database()
    #prepare_proposals_images()
    # this takes a long time -> only re-run if changed
    #generate_and_save_proposals()

    object_trainset, object_testset, object_train_loader, object_test_loader = load_and_transform_objects(
                                                                                                  batch_size=batch_size,
                                                                                                  image_resize=image_resize)

    # Display a batch of training images
    imshow_batch(object_train_loader, batch_size=16)

    # run image classification on objects
    def train(model, optimizer, train_loader, val_loader, device, epochs=10):

        criterion = nn.CrossEntropyLoss()

        out_dict = {'train_acc': [], 'val_acc': [], 'val_loss': [], 'val_loss': []}

        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            print(f"Epoch {epoch + 1}/{epochs}")

            running_loss = 0.0
            train_correct = 0
            train_total = 0
            all_preds = []
            all_labels = []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                print("Train labels:", labels)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)

                train_correct += (predicted == labels).sum().cpu().item()
                train_total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            out_dict['train_acc'].append(train_acc)
            out_dict['train_loss'].append(train_loss)

            print(f"Train loss: {train_loss:.4f}, train accuracy: {train_acc:.4f}")

            model.eval()
            running_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    print("Validation labels:", labels)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)

                    val_correct += (predicted == labels).sum().cpu().item()
                    val_total += labels.size(0)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                val_loss = running_loss / len(val_loader.dataset)
                val_acc = val_correct / train_total
                out_dict['val_acc'].append(val_acc)
                out_dict['val_loss'].append(val_loss)

            print(f"Validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Model saved with validation accuracy: {val_acc:.4f}")

        print(f"Best validation accuracy: {best_val_acc:.4f}")

        return out_dict, model

    # create model instance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    resnet18 = models.resnet18(pretrained=True)
    model = Pothole_RCNN(num_classes, resnet18).to(device)

    # hyperparameters
    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

    # out_dict, model = train(model, optimizer, object_train_loader, object_test_loader, device, epochs=epochs)
    # print(out_dict)

    torch.save(model.state_dict(), 'rcnn_model.pth')


if __name__ == "__main__":
    main()