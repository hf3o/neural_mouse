program mouse_nn
    use iso_c_binding
    implicit none

    interface
        function SDL_Init(flags) bind(C, name="SDL_Init")
            import :: c_int
            integer(c_int) :: SDL_Init
            integer(c_int), value :: flags
        end function

        subroutine SDL_Quit() bind(C, name="SDL_Quit")
        end subroutine

        function SDL_GetMouseState(x, y) bind(C, name="SDL_GetMouseState")
            import :: c_int
            integer(c_int) :: SDL_GetMouseState
            integer(c_int), intent(out) :: x, y
        end function
    end interface

    integer(c_int), parameter :: SDL_INIT_VIDEO = int(Z'00000020', c_int)
    integer(c_int) :: mx, my, prev_mx, prev_my, i, epoch
    real(8) :: x, y, x_pred, y_pred, err_x, err_y, mse
    real(8) :: learning_rate
    real(8), dimension(4) :: hidden
    real(8), dimension(4,2) :: W1 = reshape([0.2, -0.3, 0.5, -0.1,  &
                                              0.4, 0.6, -0.2, 0.3], [4,2])
    real(8), dimension(2,4) :: W2 = reshape([0.1, -0.5, 0.3, 0.7,  &
                                             -0.2, 0.4, 0.6, -0.3], [2,4])
    real(8), dimension(4) :: b1 = [0.1, -0.2, 0.05, 0.3]
    real(8), dimension(2) :: b2 = [0.05, -0.1]
    real(8), dimension(:,:), allocatable :: X_train, Y_train
    integer :: train_size

    ! Funkcja aktywacji ReLU
    contains
        function relu(x) result(y)
            real(8), intent(in) :: x
            real(8) :: y
            y = max(0.0_8, x)
        end function relu

        subroutine train_network(X_train, Y_train, train_size, learning_rate, epochs)
            integer, intent(in) :: train_size, epochs
            real(8), dimension(:,:), intent(in) :: X_train, Y_train
            real(8), dimension(4) :: hidden
            real(8), dimension(2) :: predicted
            real(8), dimension(4,2), intent(inout) :: W1
            real(8), dimension(2,4), intent(inout) :: W2
            real(8), dimension(4), intent(inout) :: b1
            real(8), dimension(2), intent(inout) :: b2
            integer :: i, j, k
            real(8) :: x, y, err_x, err_y, mse, loss, dW1(4,2), dW2(2,4), db1(4), db2(2)

            ! Trening sieci
            do epoch = 1, epochs
                mse = 0.0_8
                do i = 1, train_size
                    x = X_train(i, 1)
                    y = Y_train(i, 1)

                    ! Przepuszczenie danych przez sieć neuronową
                    hidden = matmul(W1, [x, y]) + b1
                    hidden = [relu(hidden(1)), relu(hidden(2)), relu(hidden(3)), relu(hidden(4))]
                    predicted = matmul(W2, hidden) + b2

                    ! Obliczenie błędu
                    err_x = x - predicted(1)
                    err_y = y - predicted(2)
                    mse = mse + (err_x**2 + err_y**2)

                    ! Obliczenie gradientów (backpropagation)
                    ! Gradienty dla W2 i b2
                    do j = 1, 2
                        db2(j) = -2.0_8 * (y - predicted(j))
                        do k = 1, 4
                            dW2(j,k) = -2.0_8 * (y - predicted(j)) * relu(hidden(k))
                        end do
                    end do

                    ! Gradienty dla W1 i b1
                    do j = 1, 4
                        db1(j) = -2.0_8 * (y - predicted(1)) * W2(1,j) * hidden(j) * (hidden(j) > 0)
                    end do

                    ! Aktualizacja wag i biasów
                    W2 = W2 - learning_rate * dW2
                    b2 = b2 - learning_rate * db2
                    W1 = W1 - learning_rate * dW1
                    b1 = b1 - learning_rate * db1
                end do

                mse = mse / train_size
                print *, "Epoch: ", epoch, " MSE: ", mse
            end do
        end subroutine train_network

    ! Inicjalizacja SDL
    if (SDL_Init(SDL_INIT_VIDEO) /= 0) then
        print *, "Błąd inicjalizacji SDL!"
        stop
    end if

    prev_mx = 0
    prev_my = 0
    learning_rate = 0.01
    epoch = 100
    train_size = 1000
    allocate(X_train(train_size, 2), Y_train(train_size, 2))

    ! Zbieranie danych dla treningu (wejście i wyjście)
    do i = 1, train_size
        call SDL_GetMouseState(mx, my)
        X_train(i, 1) = real(mx, 8)
        Y_train(i, 1) = real(my, 8)
    end do

    ! Trening sieci
    call train_network(X_train, Y_train, train_size, learning_rate, epoch)

    ! Zakończenie SDL
    call SDL_Quit()

end program mouse_nn
