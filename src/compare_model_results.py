from ModelsManager import ModelsManager
import matplotlib.pyplot as plt

def compare_all():
    results_path = '../results'

    model_manager = ModelsManager(results_path)
    model_names = model_manager.models

    for model_name in model_names:
        saved_model = model_manager.get_model(model_name)

        epoch, train, test = saved_model.session_stats()

        if len(epoch) > 20:
            epoch = epoch[:20]
            train = train[:20]
            test = test[:20]

        plt.subplot(1,2,1)
        plt.title('Train acc')
        plt.plot(epoch, train, label = model_name)
        plt.legend()

        plt.subplot(1,2,2)
        plt.title('Test acc')
        plt.plot(epoch, test, label = model_name)
        plt.legend()

    plt.show()

if __name__ == '__main__':
    compare_all()