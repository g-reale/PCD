#include <ncurses.h>
#include <stdlib.h>
#include <stdio.h>

int main(){
    initscr();
    printf("%d %d\n",has_colors(),can_change_color());
    endwin();
}