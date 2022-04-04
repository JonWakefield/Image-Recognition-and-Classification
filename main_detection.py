from detection import detection, image_man, animal_inference, small_inference, mic, pir, siren






# main executable file
if __name__ == "__main__":

    # begin detection
    START = input("Please press 'r' to run: ")
    if START == 'r':
        mic(1)
        state = pir(1)

        if state == 1:
            state = detection()
            print("Current state is... " , state )

            # person identified 
            if (state == 1 ):
                state = image_man( state )
                print("Human identified... activating neccessary deterents")
                # activate deterrents

            # animal identified
            if (state == 2 ):
                state = image_man( state )
                predicted = animal_inference()
                print(predicted,"identified activating neccessary deterents...")

            # small animal identified
            if (state == 3 ):
                state = image_man( state )
                predicted = small_inference()
                print(predicted,"identified activating neccessary deterents...")



