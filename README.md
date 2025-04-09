Code structure:
     P_&_N -> Dataset   -> |
         |     |           |    ->   Core.h   ->   Execute
         V     V           |          ^  ^
         Sklearn        -> |          |  |
                                      V  V
                                Linear_regression
                                     Weights
