program mouse_nn
    implicit none

    ! Deklaracje zewnętrznych funkcji SDL
    integer, external :: SDL_Init
    external :: SDL_Quit
    integer, external :: SDL_GetMouseState

    integer, parameter :: SDL_INIT_VIDEO = 32  ! Z'00000020' w decimalnym
    integer :: mx, my, prev_mx, prev_my, i, epoch
    real(8) :: learning_rate, mse
    integer :: train_size
    real(8), dimension(4) :: hidden
    real(8), dimension(4,2) :: W1
    real(8), dimension(2,4) :: W2
    real(8), dimension(4) :: b1
    real(8), dimension(2) :: b2
    real(8), dimension(:,:), allocatable :: X_train, Y_train

    ! Inicjalizacja wag i biasów
    data W1 /0.2, -0.3, 0.5, -0.1, 0.4, 0.6, -0.2, 0.3/
    data W2 /0.1, -0.5, 0.3, 0.7, -0.2, 0.4, 0.6, -0.3/
    data b1 /0.1, -0.2, 0.05, 0.3/
    data b2 /0.05, -0.1/

    ! Funkcja ReLU
    contains
        real(8) function relu(x)
            real(8), intent(in) :: x
            if (x > 0.0_8) then
                relu = x
            else
                relu = 0.0_8
            end if
        end function relu

        subroutine train_network(X_train, Y_train, train_size, learning_rate, epochs)
            integer, intent(in) :: train_size, epochs
            real(8), intent(in) :: learning_rate
            real(8), dimension(train_size,2), intent(in) :: X_train, Y_train
            real(8), dimension(4) :: hidden
            real(8), dimension(2) :: predicted
            integer :: i, j, k
            real(8) :: x, y, err_x, err_y, mse
            real(8), dimension(4,2) :: dW1
            real(8), dimension(2,4) :: dW2
            real(8), dimension(4) :: db1
            real(8), dimension(2) :: db2

            do epoch = 1, epochs
                mse = 0.0_8
                do i = 1, train_size
                    x = X_train(i, 1)
                    y = X_train(i, 2)

                    ! Feedforward
                    do j = 1, 4
                        hidden(j) = W1(j,1) * x + W1(j,2) * y + b1(j)
                        hidden(j) = relu(hidden(j))
                    end do
                    do j = 1, 2
                        predicted(j) = 0.0_8
                        do k = 1, 4
                            predicted(j) = predicted(j) + W2(j,k) * hidden(k)
                        end do
                        predicted(j) = predicted(j) + b2(j)
                    end do

                    ! Błąd
                    err_x = Y_train(i, 1) - predicted(1)
                    err_y = Y_train(i, 2) - predicted(2)
                    mse = mse + (err_x**2 + err_y**2)

                    ! Backpropagation
                    ! Gradienty dla W2 i b2
                    do j = 1, 2
                        db2(j) = -2.0_8 * (Y_train(i,j) - predicted(j))
                        do k = 1, 4
                            dW2(j,k) = db2(j) * hidden(k)
                        end do
                    end do
                    ! Gradienty dla W1 i b1
                    do j = 1, 4
                        db1(j) = 0.0_8
                        do k = 1, 2
                            db1(j) = db1(j) + db2(k) * W2(k,j)
                        end do
                        if (hidden(j) > 0.0_8) then
                            db1(j) = db1(j) * 1.0_8
                        else
                            db1(j) = 0.0_8
                        end if
                        do k = 1, 2
                            dW1(j,k) = db1(j) * X_train(i,k)
                        end do
                    end do

                    ! Aktualizacja wag i biasów
                    do j = 1, 2
                        do k = 1, 4
                            W2(j,k) = W2(j,k) - learning_rate * dW2(j,k)
                        end do
                        b2(j) = b2(j) - learning_rate * db2(j)
                    end do
                    do j = 1, 4
                        do k = 1, 2
                            W1(j,k) = W1(j,k) - learning_rate * dW1(j,k)
                        end do
                        b1(j) = b1(j) - learning_rate * db1(j)
                    end do
                end do
                mse = mse / real(train_size, 8)
                print *, "Epoch: ", epoch, " MSE: ", mse
            end do
        end subroutine train_network

    ! Główna część programu
    if (SDL_Init(SDL_INIT_VIDEO) /= 0) then
        print *, "Błąd inicjalizacji SDL!"
        stop
    end if

    prev_mx = 0
    prev_my = 0
    learning_rate = 0.01_8
    epoch = 100
    train_size = 1000
    allocate(X_train(train_size, 2))
    allocate(Y_train(train_size, 2))

    ! Zbieranie danych
    do i = 1, train_size
        mx = SDL_GetMouseState(my)
        do while (mx == prev_mx .and. my == prev_my)
            mx = SDL_GetMouseState(my)
        end do
        X_train(i, 1) = real(mx, 8)
        X_train(i, 2) = real(my, 8)
        Y_train(i, 1) = real(mx, 8)  ! Wyjście to ta sama pozycja
        Y_train(i, 2) = real(my, 8)
        prev_mx = mx
        prev_my = my
    end do

    ! Trening sieci
    call train_network(X_train, Y_train, train_size, learning_rate, epoch)

    ! Zakończenie
    call SDL_Quit()
    deallocate(X_train)
    deallocate(Y_train)

end program mouse_nn
