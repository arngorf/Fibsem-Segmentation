from ModelsManager import ModelsManager
import matplotlib.pyplot as plt
import numpy as np

model_manager = None

def compare_all():
    results_path = '../results'

    model_manager = ModelsManager(results_path)
    model_names = model_manager.models

    preprocessing = ['rot', 'fov', 'aff', 'warp', 'noi']

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
            #plt.plot(epoch, train, label = model_name+'.'+session_name+'_train_acc')
            #plt.legend()

            #plt.subplot(1,2,2)
            #plt.title('Test acc')
            printed_name = model_name.replace('conv_2_layer_conf_', '')

            printed_name = printed_name.replace('rotation', 'rot')
            printed_name = printed_name.replace('foveation', 'fov')
            printed_name = printed_name.replace('linear_deformation', 'aff')
            printed_name = printed_name.replace('non_linear_resampling', 'warp')
            printed_name = printed_name.replace('noise', 'noi')

            for i in range(5):
                plt.subplot(1,5,i+1)
                plt.title(preprocessing[i])
                color = 'black'
                print(printed_name)
                if preprocessing[i] == printed_name:
                    #color = 'red'
                    plt.plot(epoch, test, color=color)
                plt.ylim(0,1)

            #plt.legend()

    plt.show()

def plot_single_all_sessions(model_name):
    results_path = '../results'

    model_manager = ModelsManager(results_path)

    saved_model = model_manager.get_model(model_name)

    for session_name in saved_model.sessions:
        epoch, train, test = saved_model.session_stats(session_name)

        plt.title(model_name)

        plt.plot(epoch, test, label=session_name)
        plt.ylim(0,1)

    plt.legend()

def plot_single(saved_model, session_name='default'):

    epoch, train, test = saved_model.session_stats(session_name)
    model_name = saved_model.name
    formated_name = model_name.replace('conv_2_layer_', '').replace('_','\n')
    color = 'black'
    if 'rot' in formated_name:
        formated_name = 'rot'
        color = 'red'
    if 'fovea' in formated_name:
        formated_name = 'fovea'
        color = 'blue'
    if 'linear' in formated_name:
        formated_name = 'linear'
        color = 'green'

    plt.plot(epoch, train, color=color, label=formated_name)
    plt.plot(epoch, test, color=color, ls='--', label=formated_name)

def plot_specific(model_names, session_name='default'):

    global model_manager

    if model_manager == None:
        results_path = '../results'
        model_manager = ModelsManager(results_path)

    for model_name in model_names:
        if model_manager.has_model(model_name):
            saved_model = model_manager.get_model(model_name)
            plot_single(saved_model, session_name)

def plot_bar_chart_average_success(model_names, session_name='default'):

    global model_manager

    if model_manager == None:
        results_path = '../results'
        model_manager = ModelsManager(results_path)

    y_avg = []
    y_max = []
    formated_names = []

    for model_name in model_names:
        if model_manager.has_model(model_name):

            saved_model = model_manager.get_model(model_name)

            epoch, train, test = saved_model.session_stats(session_name)

            avg_acc = np.mean(test)
            max_acc = np.max(test)

            y_avg.append(avg_acc)
            y_max.append(max_acc)

            formated_name = model_name.replace('conv_2_layer_', '').replace('_','\n')
            formated_names.append(formated_name)

    x = range(1,len(formated_names) + 1)

    plt.plot(x, y_avg, 'go', label='average')
    plt.plot(x, y_max, 'ro', label='max')
    plt.xticks(x, formated_names) #, rotation='vertical'
    plt.ylabel('average test accuracy')
    plt.subplots_adjust(bottom=0.25)
    plt.legend()

if __name__ == '__main__':
    #compare_all()


    all_model_names = []

    plot_num = 0

    for conv_prob in ['0.35', '0.5']:
        for dense_prob in ['0.35', '0.5', '0.65']:

            plot_num += 1
            plt.subplot(2, 3, plot_num)

            model_names = []

            for pp in ['rot', 'fovea', 'linear']:
                name = 'conv_2_layer_' + pp + '_' + conv_prob + '_' + dense_prob
                model_names.append(name)
                all_model_names.append(name)

            plot_specific(model_names)
            plt.title('CDP: ' + conv_prob + ', DDP: ' + dense_prob)
            plt.ylim(0.5,1)
            plt.legend()

    plt.show()

    plot_bar_chart_average_success(all_model_names)
    plt.ylim(0.75,1)

    plt.show()