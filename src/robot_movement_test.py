import numpy as np
import matplotlib.pyplot as plt
import math
import os

from intpy.intpy import initialize_intpy, deterministic

path = "out/"

iteration = np.linspace(1, 20, 20)
iteration = iteration[np.newaxis, :]

#NÃO É DETERMINÍSTICA PORQUE LÊ DADOS DE UM ARQUIVO EXTERNO (open(filename))
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

#É DETERMINÍSTICA (SÓ É NECESSÁRIO VERIFICAR SE OCORRERÃO PROBLEMAS PELO RETORNO UTILIZAR
#FUNÇÕES DO MÓDULO NUMPY)
@deterministic
def typeConverter(data, desiredType):
    """
    :param data: data to be converted
    :param desiredType: to which the conversion is desired
    :return: returns the converted data
    """
    newList = [desiredType(x) for x in data]
    return np.asarray(newList)

#NÃO É DETERMINÍSTICA PORQUE GERA UM GRÁFICO E O SALVA (plt.savefig)
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
    plt.figure(figsize=(15, 15))
    plt.scatter(qty1, qty2)
    plt.title(title)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # path = desired path

    plt.savefig(path + filename)

#NÃO É DETERMINÍSTICA PORQUE GERA UM GRÁFICO E O SALVA (plt.savefig)
def finalPlot(finaltitle, filename,
              x_left, y_left,
              x_final, y_final,
              x_right, y_right):
    """
    :param finaltitle: title for the plot
    :param filename: filename for storage
    :param x_left: x coordinates for left wheel
    :param y_left: y coordinates for left wheel
    :param x_final: x coordinates for robot center
    :param y_final: y coordinates for robot center
    :param x_right: x coordinates for right wheel
    :param y_right: y coordinates for right wheel
    :return: plot depicting the final robot and wheel positions with respect to the robot starting position
    """
    plt.figure(figsize=(15, 15))
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.title(finaltitle)
    plt.scatter(x_left, y_left, c='r', label='left marker')
    plt.scatter(x_final, y_final, c='b', label='robot location')
    plt.scatter(x_right, y_right, c='g', label='right marker')
    plt.plot(31.4, 6.65, 'kx', label='startpoint')
    plt.xlabel('cms')
    plt.ylabel('cms')
    plt.legend()
    plt.grid()
    # path = desired path
    plt.savefig(path + filename)

#NÃO É DETERMINÍSTICA PORQUE CHAMA FUNÇÕES NÃO DETERMINÍSTICAS (plotValues)
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
###############################ESSA PARTE DETERMINÍSTICA
        @deterministic
        def f1(x_left_coordinates, y_left_coordinates, x_right_coordinates, y_right_coordinates):
            X_final_pose_straight = np.divide(x_left_coordinates + x_right_coordinates, 2)
            Y_final_pose_straight = np.divide(y_left_coordinates + y_right_coordinates, 2)

            slope_coordinate_wheel_straight = [x / y for x, y in zip((y_right_coordinates - y_left_coordinates),
                                                                    (x_right_coordinates - x_left_coordinates))]

            arctan_Values_straight = abs(np.rad2deg(np.arctan(slope_coordinate_wheel_straight))) + np.rad2deg(np.pi / 2)
            return X_final_pose_straight, Y_final_pose_straight, arctan_Values_straight
        X_final_pose_straight, Y_final_pose_straight, arctan_Values_straight = f1(x_left_coordinates, y_left_coordinates, x_right_coordinates, y_right_coordinates)
#############################################################

        plotValues(arctan_Values_straight, iteration,
                   'Variation of robot heading for Straight motion test',
                   'Angle (in°) of perpendicular to the axle with x axis', 'Iteration Number',
                   'Variation of robot heading for Straight motion test.pdf')

        finalPlot('Final Pose Plot for Straight Motion',
                  'Final pose plot for Straight Motion.pdf',
                  x_left_coordinates,
                  y_left_coordinates,
                  X_final_pose_straight,
                  Y_final_pose_straight,
                  x_right_coordinates,
                  y_right_coordinates)
        return X_final_pose_straight, Y_final_pose_straight

    elif pose == 'left':
###############################ESSA PARTE DETERMINÍSTICA
        @deterministic
        def f2(x_left_coordinates, y_left_coordinates, x_right_coordinates, y_right_coordinates):
            X_final_pose_left = np.divide(x_left_coordinates + x_right_coordinates, 2)
            Y_final_pose_left = np.divide(y_left_coordinates + y_right_coordinates, 2)

            slope_coordinate_wheel_left = [x / y for x, y in zip((y_right_coordinates - y_left_coordinates),
                                                                (x_right_coordinates - x_left_coordinates))]
            arctan_Values_left = abs(np.rad2deg(np.arctan(slope_coordinate_wheel_left))) + np.rad2deg(np.pi / 2)
            return X_final_pose_left, Y_final_pose_left, arctan_Values_left
        X_final_pose_left, Y_final_pose_left, arctan_Values_left = f2(x_left_coordinates, y_left_coordinates, x_right_coordinates, y_right_coordinates)
#######################################

        plotValues(arctan_Values_left, iteration,
                   'Variation of robot heading for Left motion test',
                   'Angle (in°) of perpendicular to the axle with x axis', 'Iteration Number',
                   'Variation of robot heading for Left motion test.pdf')

        finalPlot('Final Pose Plot for Left Motion',
                  'Final pose plot for Left Motion.pdf',
                  x_left_coordinates,
                  y_left_coordinates,
                  X_final_pose_left,
                  Y_final_pose_left,
                  x_right_coordinates,
                  y_right_coordinates)
        return X_final_pose_left, Y_final_pose_left

    elif pose == 'right':
###############################ESSA PARTE DETERMINÍSTICA
        @deterministic
        def f3(x_left_coordinates, y_left_coordinates, x_right_coordinates, y_right_coordinates):
            X_final_pose_right = np.divide(x_left_coordinates + x_right_coordinates, 2)
            Y_final_pose_right = np.divide(y_left_coordinates + y_right_coordinates, 2)

            slope_coordinate_wheel_right = [x / y for x, y in zip((y_right_coordinates - y_left_coordinates),
                                                                (x_right_coordinates - x_left_coordinates + 1e-7))]

            arctan_Values_right = np.rad2deg(np.arctan(slope_coordinate_wheel_right))

            for i in range(len(arctan_Values_right)):
                if arctan_Values_right[i] < 0:
                    arctan_Values_right[i] = np.rad2deg(np.arctan(slope_coordinate_wheel_right[i])) + np.rad2deg(np.pi / 2)

                elif (arctan_Values_right[i]) > 0:
                    arctan_Values_right[i] = np.rad2deg(np.arctan(slope_coordinate_wheel_right[i])) - np.rad2deg(np.pi / 2)
            return X_final_pose_right, Y_final_pose_right, arctan_Values_right
        X_final_pose_right, Y_final_pose_right, arctan_Values_right = f3(x_left_coordinates, y_left_coordinates, x_right_coordinates, y_right_coordinates)
################################################

        plotValues(arctan_Values_right, iteration,
                   'Variation of robot heading for Right motion test',
                   'Angle (in°) of perpendicular to the axle with x axis', 'Iteration Number',
                   'Variation of robot heading for Right motion test.pdf')

        finalPlot('Final Pose Plot for Right Motion',
                  'Final pose plot for Right Motion.pdf',
                  x_left_coordinates,
                  y_left_coordinates,
                  X_final_pose_right,
                  Y_final_pose_right,
                  x_right_coordinates,
                  y_right_coordinates)
        return X_final_pose_right, Y_final_pose_right

#NÃO É DETERMINÍSTICA PORQUE CHAMA FUNÇÕES NÃO DETERMINÍSTICAS (finalPlot)
def finalVisualization(x_left_wheel_right, y_left_wheel_right, x_left_wheel_left, y_left_wheel_left,
                       x_left_wheel_straight, y_left_wheel_straight,
                       x_right_wheel_right, y_right_wheel_right, x_right_wheel_left, y_right_wheel_left,
                       x_right_wheel_straight, y_right_wheel_straight,
                       x_center_right, y_center_right, x_center_left, y_center_left,
                       x_center_straight, y_center_straight):
#####################ESSA PARTE É DETERMINÍSTICA
    @deterministic
    def f4(x_right_wheel_right, x_right_wheel_left, x_right_wheel_straight,
           y_right_wheel_right, y_right_wheel_left, y_right_wheel_straight,
           x_left_wheel_right, x_left_wheel_left, x_left_wheel_straight,
           y_left_wheel_right, y_left_wheel_left, y_left_wheel_straight,
           x_center_right, x_center_left, x_center_straight,
           y_center_right, y_center_left, y_center_straight):
        X_right_wheel = np.hstack(
            (x_right_wheel_right, x_right_wheel_left, x_right_wheel_straight))
        Y_right_wheel = np.hstack(
            (y_right_wheel_right, y_right_wheel_left, y_right_wheel_straight))

        X_left_wheel = np.hstack(
            (x_left_wheel_right, x_left_wheel_left, x_left_wheel_straight))
        Y_left_wheel = np.hstack(
            (y_left_wheel_right, y_left_wheel_left, y_left_wheel_straight))

        X_center = np.hstack(
            (x_center_right, x_center_left, x_center_straight))
        Y_center = np.hstack(
            (y_center_right, y_center_left, y_center_straight))
        return X_right_wheel, Y_right_wheel, X_left_wheel, Y_left_wheel, X_center, Y_center
    X_right_wheel, Y_right_wheel, X_left_wheel, Y_left_wheel, X_center, Y_center = f4(x_right_wheel_right, x_right_wheel_left, x_right_wheel_straight,
                                                                                      y_right_wheel_right, y_right_wheel_left, y_right_wheel_straight,
                                                                                      x_left_wheel_right, x_left_wheel_left, x_left_wheel_straight,
                                                                                      y_left_wheel_right, y_left_wheel_left, y_left_wheel_straight,
                                                                                      x_center_right, x_center_left, x_center_straight,
                                                                                      y_center_right, y_center_left, y_center_straight)
############################################
    finalPlot('Final Pose plot',
              'Final Pose plot.pdf',
              X_left_wheel, Y_left_wheel,
              X_center, Y_center,
              X_right_wheel, Y_right_wheel)

#NÃO É DETERMINÍSTICA PORQUE CHAMA FUNÇÕES NÃO DETERMINÍSTICA(finalPoseEstimation)
@initialize_intpy(__file__)
def main():
    # STRAIGHT MOTION
    temp = readData('Readings_robot_motion/straightWheelPositions.csv')[0][36:]
    X_left_wheel_straight = temp.split(',')
    X_left_wheel_straight_new = typeConverter(X_left_wheel_straight, float)

    temp = readData('Readings_robot_motion/straightWheelPositions.csv')[1][36:]
    Y_left_wheel_straight = temp.split(',')
    Y_left_wheel_straight_new = typeConverter(Y_left_wheel_straight, float)

    temp = readData('Readings_robot_motion/straightWheelPositions.csv')[2][36:]
    X_right_wheel_straight = temp.split(',')
    X_right_wheel_straight_new = typeConverter(X_right_wheel_straight, float)

    temp = readData('Readings_robot_motion/straightWheelPositions.csv')[3][36:]
    Y_right_wheel_straight = temp.split(',')
    Y_right_wheel_straight_new = typeConverter(Y_right_wheel_straight, float)

    x_final_pose_straight, y_final_pose_straight = finalPoseEstimation('straight', X_left_wheel_straight_new,
                                                                       Y_left_wheel_straight_new,
                                                                       X_right_wheel_straight_new,
                                                                       Y_right_wheel_straight_new)
    # LEFT MOTION
    temp = readData('Readings_robot_motion/leftWheelPositions.csv')[0][32:]
    X_left_wheel_left = temp.split(',')
    X_left_wheel_left_new = typeConverter(X_left_wheel_left, float)

    temp = readData('Readings_robot_motion/leftWheelPositions.csv')[1][32:]
    Y_left_wheel_left = temp.split(',')
    Y_left_wheel_left_new = typeConverter(Y_left_wheel_left, float)

    temp = readData('Readings_robot_motion/leftWheelPositions.csv')[2][32:]
    X_right_wheel_left = temp.split(',')
    X_right_wheel_left_new = typeConverter(X_right_wheel_left, float)

    temp = readData('Readings_robot_motion/leftWheelPositions.csv')[3][32:]
    Y_right_wheel_left = temp.split(',')
    Y_right_wheel_left_new = typeConverter(Y_right_wheel_left, float)

    x_final_pose_left, y_final_pose_left = finalPoseEstimation('left', X_left_wheel_left_new, Y_left_wheel_left_new,
                                                               X_right_wheel_left_new, Y_right_wheel_left_new)

    # RIGHT MOTION
    temp = readData('Readings_robot_motion/rightWheelPositions.csv')[0][33:]
    X_left_wheel_right = temp.split(',')
    X_left_wheel_right_new = typeConverter(X_left_wheel_right, float)

    temp = readData('Readings_robot_motion/rightWheelPositions.csv')[1][33:]
    Y_left_wheel_right = temp.split(',')
    Y_left_wheel_right_new = typeConverter(Y_left_wheel_right, float)

    temp = readData('Readings_robot_motion/rightWheelPositions.csv')[2][33:]
    X_right_wheel_right = temp.split(',')
    X_right_wheel_right_new = typeConverter(X_right_wheel_right, float)

    temp = readData('Readings_robot_motion/rightWheelPositions.csv')[3][33:]
    Y_right_wheel_right = temp.split(',')
    Y_right_wheel_right_new = typeConverter(Y_right_wheel_right, float)

    x_final_pose_right, y_final_pose_right = finalPoseEstimation('right', X_left_wheel_right_new,
                                                                 Y_left_wheel_right_new,
                                                                 X_right_wheel_right_new, Y_right_wheel_right_new)
    finalVisualization(X_left_wheel_right_new,
                       Y_left_wheel_right_new,
                       X_left_wheel_left_new,
                       Y_left_wheel_left_new,
                       X_left_wheel_straight_new,
                       Y_left_wheel_straight_new,
                       X_right_wheel_right_new,
                       Y_right_wheel_right_new,
                       X_right_wheel_left_new,
                       Y_right_wheel_left_new,
                       X_right_wheel_straight_new,
                       Y_right_wheel_straight_new,
                       x_final_pose_right,
                       y_final_pose_right,
                       x_final_pose_left,
                       y_final_pose_left,
                       x_final_pose_straight,
                       y_final_pose_straight)


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    main()
    print(time.perf_counter()-start)
