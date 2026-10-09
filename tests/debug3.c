#include "board.h"
#include <stdio.h>
int main(void){
    Board b; board_init(&b,9);
    for(int x=0;x<9;x++)for(int y=0;y<9;y++){
        int inner=(x>=3&&x<=5&&y>=3&&y<=5);
        if(!inner) b.cell[y*9+x]=1;
    }
    b.to_move=1; b.nmoves=80; b.passes=0;
    board_play(&b, 3*9+3);              /* black (3,3) */
    printf("after black (3,3): to_move=%d nmoves=%d score=%+.1f\n", b.to_move, b.nmoves, board_score(&b,7.0));
    Rng rng; rng_seed(&rng,99);
    Board u=b;
    int steps=0;
    while(u.passes<2 && u.nmoves<b.nmoves+243){
        int m=board_random_move(&u,&rng,0);
        char s[8]; board_vertex(&u,m,s,8);
        if(!board_play(&u,m)){printf("illegal %s\n",s);break;}
        steps++;
        if(steps<25) printf("  step %2d: %s -> to_move %d passes %d score %+.1f\n",steps,s,u.to_move,u.passes,board_score(&u,7.0));
    }
    printf("steps=%d passes=%d nmoves=%d score=%+.1f winner=%d\n",steps,u.passes,u.nmoves,board_score(&u,7.0),board_winner(&u,7.0));
    printf("playout value from white's view: %+.2f\n", board_playout_value(&b,&rng,243,7.0));
    return 0;
}
