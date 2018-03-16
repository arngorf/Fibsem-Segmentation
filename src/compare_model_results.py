from ModelsManager import ModelsManager
import matplotlib.pyplot as plt

def compare_all():
    results_path = '../results'

    model_manager = ModelsManager(results_path)
    model_names = model_manager.models

    print(model_names)

    for model_name in model_names:
        saved_model = model_manager.get_model(model_name)

        for session_name in  saved_model.sessions:
            #if session_name == 'default':
            #    continue
            epoch, train, test = saved_model.session_stats(session_name)

            '''if len(epoch) > 32:
                epoch = epoch[:32]
                train = train[:32]
                test = test[:32]'''

            #plt.subplot(1,2,1)
            #plt.title('Train acc')
            plt.plot(epoch, train, label = model_name+'.'+session_name+'_train_acc')
            #plt.legend()

            #plt.subplot(1,2,2)
            #plt.title('Test acc')
            plt.plot(epoch, test, label = model_name+'_test_acc')
            plt.legend()

    plt.show()

if __name__ == '__main__':
    compare_all()