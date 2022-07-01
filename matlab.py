import matplotlib.pyplot as plt
import time



def plt_all(plt_dot_x,plt_dot_y):
    plt.text(311,242,"Right")
    plt.fill([311,521,311,521],[242,242,-58,-58],       color="g",alpha=0.5)
    plt.scatter(416, 92, color="r",alpha=0.2)

    plt.text(-118,242,"Left")
    plt.fill([-118,92,-118,92],[242,242,-58,-58],     color="g",alpha=0.5)
    plt.scatter(-13, 92, color="r",alpha=0.2)

    # plt.text(-204,556,"Puzzle")
    # plt.fill([-204,13,13,-204],[556,556,276,276],     color="gray",alpha=0.5)
    # plt.scatter(-95, 394, color="r",alpha=0.2)

    # plt.text(-204,276,"Correction")
    # plt.fill([-204,13,13,-204],[276,276,166,166],     color="black",alpha=0.5)
    # plt.scatter(-95, 221, color="r",alpha=0.2)

    plt.scatter(plt_dot_x, plt_dot_y, color="b",alpha=1)


def plt_all_2(Puzzle):
    plt.text(13,512,"Right")
    plt.fill([13,223,223,13],[512,512,218,218],       color="g",alpha=0.5)
    plt.scatter(-307, 365, color="r",alpha=0.2)

    plt.text(-414,512,"Left")
    plt.fill([-414,-204,-204,-414],[512,512,218,218],     color="g",alpha=0.5)
    plt.scatter(118, 365, color="r",alpha=0.2)

    plt.text(-204,556,"Puzzle")
    plt.fill([-204,13,13,-204],[556,556,276,276],     color="gray",alpha=0.5)
    plt.scatter(-95, 394, color="r",alpha=0.2)

    plt.text(-204,276,"Correction")
    plt.fill([-204,13,13,-204],[276,276,166,166],     color="black",alpha=0.5)
    plt.scatter(-95, 221, color="r",alpha=0.2)

    for i in range(len(Puzzle)):
        if Puzzle[i][2]==0:
            plt_dot_x = Puzzle[i][0][0]
            plt_dot_y = Puzzle[i][0][1]
            plt.scatter(plt_dot_x, plt_dot_y    ,color="b",alpha=1)
            plt.text(plt_dot_x+3,plt_dot_y,i+1    ,color="black")
        elif Puzzle[i][2]>0:
            plt_dot_x = Puzzle[i][0][0]
            plt_dot_y = Puzzle[i][0][1]
            plt.scatter(plt_dot_x, plt_dot_y, color="r",alpha=1)
            plt.text(plt_dot_x,plt_dot_y,i+1)


def plt_all_3(Puzzle,sucker2camera):
    plt.text(311,242,"Right")
    plt.fill([311,521,521,311],[242,242,-58,-58],       color="g",alpha=0.5)
    plt.scatter(416, 92, color="r",alpha=0.2)

    plt.text(-118,242,"Left")
    plt.fill([-118,92,92,-118],[242,242,-58,-58],     color="g",alpha=0.5)
    plt.scatter(-13, 92, color="r",alpha=0.2)

    # plt.text(-204,556,"Puzzle")
    # plt.fill([-204,13,13,-204],[556,556,276,276],     color="gray",alpha=0.5)
    # plt.scatter(-95, 394, color="r",alpha=0.2)

    # plt.text(-204,276,"Correction")
    # plt.fill([-204,13,13,-204],[276,276,166,166],     color="black",alpha=0.5)
    # plt.scatter(-95, 221, color="r",alpha=0.2)

    for i in range(len(Puzzle)):
        if Puzzle[i][2]==0:
            plt_dot_x = Puzzle[i][0][0]-sucker2camera[0]
            plt_dot_y = Puzzle[i][0][1]-sucker2camera[1]
            plt.scatter(plt_dot_x, plt_dot_y    ,color="b",alpha=1)
            plt.text(plt_dot_x+3,plt_dot_y,i+1    ,color="black")
        elif Puzzle[i][2]>0:
            plt_dot_x = Puzzle[i][0][0]-sucker2camera[0]
            plt_dot_y = Puzzle[i][0][1]-sucker2camera[1]
            plt.scatter(plt_dot_x, plt_dot_y, color="r",alpha=1)
            plt.text(plt_dot_x,plt_dot_y,i+1)

    # plt.scatter(plt_dot_x, plt_dot_y, color="b",alpha=1)

def plt_(Puzzle,sucker2camera):

    # plt.figure(figsize=(16,10))
    # plt.xlim(-415, 225)
    # plt.ylim(166, 556)

    plt.xlim(-118, 521)
    plt.ylim(-98, 294)
    
    #plt.figure(figsize=(8,8))

    plt_all_3(Puzzle,sucker2camera)
    
    plt.show(block=False)
    plt.pause(1)
    # plt.close()
    # plt.pause(0.033)



if __name__ == "__main__":

    plt_dot_x = [0]
    plt_dot_y = [0]


    while 1:
        
        plt.xlim(-10, 630)
        plt.ylim(-20, 379)
        
        x= input("enter x: ")
        y= input("enter y: ")
        plt_dot_x.append(x)
        plt_dot_y.append(y)
        plt_all_3(plt_dot_x,plt_dot_y)
        
        # plt.show()
        plt.pause(0.033)