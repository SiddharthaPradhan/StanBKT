import arviz as az
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Start")
    # data = pickle.load(open('test.pickle','rb'))
    data = pickle.load(open("./test.pkl", "rb"))
    axes = az.plot_trace(data, compact=False, backend='matplotlib', show=False)
    fig = axes.ravel()[0].figure
    fig.tight_layout()
    fig.show(blocking=True)
# return fig
#     plt.tight_layout()
#     plt.show()

    
    print("Done..")    