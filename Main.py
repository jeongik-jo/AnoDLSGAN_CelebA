import HyperParameters as hp
import Train
import Models
import Dataset


def main():
    train_dataset, id_dataset = Dataset.load_id_dataset()
    ood_datasets = Dataset.load_ood_datasets(id_dataset)

    encoder = Models.Encoder()
    decoder = Models.Decoder()
    svm = Models.Svm()

    if hp.load_model:
        encoder.load()
        decoder.load()
        svm.load()

    Train.train(encoder, decoder, svm, train_dataset, id_dataset, ood_datasets)


main()
