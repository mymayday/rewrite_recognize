from train import train
from test  import test
from config import configer

def main():
    train(configer)
    test(configer)
    #gen_out_excel(configer)

if __name__ == "__main__":
    main()
