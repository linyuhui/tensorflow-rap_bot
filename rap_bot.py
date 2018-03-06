import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import train


def main():
    train.train(2)


if __name__ == "__main__":
    main()
