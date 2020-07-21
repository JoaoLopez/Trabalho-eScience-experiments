import numpy as np
import matplotlib.pyplot as plt
import math

iteration = np.linspace(0, 20, 20)
iteration = iteration[np.newaxis, :]


def readData(filename):
    """
    :param filename: file from which the data is to be input
    :return: coordinate in the form of list
    """
    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content


def typeConverter(data, desiredType):
    """
    :param data: data to be converted
    :param desiredType: to which the conversion is desired
    :return: returns the converted data
    """
    newList = [desiredType(x) for x in data]
    return np.asarray(newList)


def plotValues(qty1, qty2, title, xlabel, ylabel, filename):
    """
    :param qty1: x value
    :param qty2: y value
    :param title: title of plot
    :param xlabel: xlabel of plot
    :param ylabel: ylabel of plot
    :param filename: filename of the resulting plot
    :return: scatter plot
    """
    plt.scatter(qty1, qty2)
    plt.title(title)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    # plt.scatter(arctan_Values_straight, iteration)
    # plt.title('Variation of robot heading for Straight motion test')
    # plt.xlabel('Angle (in°)')
    # plt.ylabel('Iteration Number')
    # plt.savefig('Variation of robot heading for Straight motion test.pdf')


def finalPoseEstimation(pose, x_left_coordinates, y_left_coordinates, x_right_coordinates, y_right_coordinates):
    """
    :param pose: straight, left or right
    :param x_left_coordinates: x coordinates for the left wheel
    :param y_left_coordinates: y coordinates for the left wheel
    :param x_right_coordinates: x coordinates for the right wheel
    :param y_right_coordinates: y coordinates for the right wheel
    :return: coordinate estimate for the final pose
    """
    if pose == 'straight':
        X_final_pose_straight = np.divide(x_left_coordinates + x_right_coordinates, 2)
        Y_final_pose_straight = np.divide(y_left_coordinates + y_right_coordinates, 2)

        slope_coordinate_wheel_straight = [x/y for x, y in zip((y_right_coordinates - y_left_coordinates), (x_right_coordinates - x_left_coordinates))]

        arctan_Values_straight = abs(np.rad2deg(np.arctan(slope_coordinate_wheel_straight))) + np.rad2deg(np.pi/2)

        plotValues(arctan_Values_straight, iteration,
                   'Variation of robot heading for Straight motion test',
                   'Angle (in°)', 'Iteration Number',
                   'Variation of robot heading for Straight motion test.pdf')


def main():
    X_left_wheel_straight = readData('straightWheelPositions.txt')[0][36:].split(',')
    X_left_wheel_straight_new = typeConverter(X_left_wheel_straight, float)

    Y_left_wheel_straight = readData('straightWheelPositions.txt')[1][36:].split(',')
    Y_left_wheel_straight_new = typeConverter(Y_left_wheel_straight, float)

    X_right_wheel_straight = readData('straightWheelPositions.txt')[2][36:].split(',')
    X_right_wheel_straight_new = typeConverter(X_right_wheel_straight, float)

    Y_right_wheel_straight = readData('straightWheelPositions.txt')[3][36:].split(',')
    Y_right_wheel_straight_new = typeConverter(Y_right_wheel_straight, float)

    finalPoseEstimation('straight', X_left_wheel_straight_new, Y_left_wheel_straight_new,
                        X_right_wheel_straight_new, Y_right_wheel_straight_new)


if __name__ == '__main__':
    main()






