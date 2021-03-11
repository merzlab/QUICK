#include "util.fh"
!
!	greedy_distrubute.f90
!	new_quick
!
!	Created by Yipu Miao on 2/23/11.
!	Copyright 2011 University of Florida. All rights reserved.
!

!-----------------------------------------------------------
! greedy_distrubutie
!-----------------------------------------------------------
! Yipu Miao 11/19/2010
! greedy algrithm to obtain optimized distrubution
!-----------------------------------------------------------    
    subroutine greedy_distrubute(j,n,node,node_jn,node_j)
    implicit none
    integer n               ! no. of stuff
    integer j(n)            ! Value of n stuff
    integer jorder(n)       ! value list after sort
    integer node            ! no. of nodes
    integer node_jn(0:node-1)   ! total no. of stuff one node takes
    integer node_j(0:node-1,n)  ! the nth stuff one node takes
    integer i,jn,k,jj,ii,maxnode,minnode
    integer max_val,min_val
    integer node_jtot(0:node-1) ! total value of one node
    logical isUsed(n)

    do i=0,node-1
        node_jn(i)=0
        node_jtot(i)=0
    enddo
    do i=1,n
        isUsed(i)=.false.
    enddo
  
    minnode=0
    ! The basis idea is to put the largest element to the group with fewest 
    ! value
    do i=1,n
        ! first find the most valuable element
        max_val=0
        do jj=1,n
           if((j(jj).ge.max_val).and.(.not.isUsed(jj))) then
              ii=jj
              max_val=j(jj)
           endif
        enddo
        isUsed(ii)=.true.

        ! then put it to the group with minimum value
        node_jn(minnode)=node_jn(minnode)+1
        node_j(minnode,node_jn(minnode))=ii
        node_jtot(minnode)=node_jtot(minnode)+j(ii)
        
        ! find now which group is the most valueless
        do jn=0,node-1
            if (node_jtot(jn).lt.node_jtot(minnode)) minnode=jn
        enddo
    enddo

    end subroutine greedy_distrubute
    
    