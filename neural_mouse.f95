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
    integer(c_int) :: mx, my, prev_mx, prev_my, i
    real(8) :: x, y, x_pred, y_pred, err_x, err_y
    real(8), dimension(4) :: hidden
    real(8), dimension(4,2) :: W1 = reshape([0.2, -0.3, 0.5, -0.1,  &
                                              0.4, 0.6, -0.2, 0.3], [4,2])
    real(8), dimension(2,4) :: W2 = reshape([0.1, -0.5, 0.3, 0.7,  &
                                             -0.2, 0.4, 0.6, -0.3], [2,4])
    real(8), dimension(4) :: b1 = [0.1, -0.2, 0.05, 0.3]
    real(8), dimension(2) :: b2 = [0.05, -0.1]

    ! Funkcja aktywacji ReLU
    contains
        function relu(x) result(y)
            real(8), intent(in) :: x
            real(8) :: y
            y = max(0.0_8, x)
        end function relu

    ! Inicjalizacja SDL
    if (SDL_Init(SDL_INIT_VIDEO) /= 0) then
        print *, "Błąd inicjalizacji SDL!"
        stop
    end if

    prev_mx = 0
    prev_my = 0

    do i = 1, 100
        ! Pobranie aktualnej pozycji myszy
        call SDL_GetMouseState(mx, my)
        x = real(mx, 8)
        y = real(my, 8)

        ! Przepuszczenie danych przez sieć neuronową
        hidden = matmul(W1, [x, y]) + b1
        hidden = [relu(hidden(1)), relu(hidden(2)), relu(hidden(3)), relu(hidden(4))]
        x_pred = dot_product(W2(1,:), hidden) + b2(1)
        y_pred = dot_product(W2(2,:), hidden) + b2(2)

        ! Obliczenie błędu
        err_x = x - x_pred
        err_y = y - y_pred

        ! Wypisanie wyników
        print "(A, 2F10.2, A, 2F10.2, A, 2F10.2)", "Rzeczywiste: X=", x, " Y=", y, &
              " | Predykcja: X'=", x_pred, " Y'=", y_pred, " | Błąd: ΔX=", err_x, " ΔY=", err_y

        prev_mx = mx
        prev_my = my

        ! Opóźnienie 0.1s
        call sleep(1)
    end do

    ! Zakończenie SDL
    call SDL_Quit()

end program mouse_nn

