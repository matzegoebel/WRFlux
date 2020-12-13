!WRF:MODEL_LAYER:DYNAMICS
!
MODULE module_avgflx_em

  USE module_bc
  USE module_model_constants
  USE module_wrf_error
  USE module_state_description

CONTAINS

!-------------------------------------------------------------------------------


  subroutine zero_avgflx(avgflx_rum,avgflx_rvm,avgflx_wwm, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte,           &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1, &
       & config_flags,                                                                           &
       & t_mean,tx_mean,ty_mean,tz_mean,tt_mean,ftx_adv_mean,fty_adv_mean,ftz_adv_mean, &
       & q_mean,qx_mean,qy_mean,qz_mean,qq_mean,fqx_adv_mean,fqy_adv_mean,fqz_adv_mean,          &
       & u_mean,ux_mean,uy_mean,uz_mean,fux_adv_mean,fuy_adv_mean,fuz_adv_mean,                  &
       & v_mean,vx_mean,vy_mean,vz_mean,fvx_adv_mean,fvy_adv_mean,fvz_adv_mean,                  &
       & ww_mean,wd_mean,zwind_mean,wx_mean,wy_mean,wz_mean,fwx_adv_mean,fwy_adv_mean,fwz_adv_mean,  &
       & ftx_corr,fty_corr,corr_dtdt,                                                           &
       & fqx_corr,fqy_corr,corr_dqdt,                                                              &
       & fux_corr,fuy_corr,corr_dudt,                                                              &
       & fvx_corr,fvy_corr,corr_dvdt,                                                              &
       & fwx_corr,fwy_corr,corr_dwdt,                                                              &
       & ftx_adv_mean_2nd,fty_adv_mean_2nd,ftz_adv_mean_2nd,ftz_adv_mean_prog,ftz_adv_mean_2nd_prog, &
       & ftx_corr_2nd,fty_corr_2nd,corr_dtdt_2nd,                                                    &
       & fqx_adv_mean_2nd,fqy_adv_mean_2nd,fqz_adv_mean_2nd,fqz_adv_mean_prog,fqz_adv_mean_2nd_prog, &
       & fqx_corr_2nd,fqy_corr_2nd,corr_dqdt_2nd,                                                    &
       & fux_adv_mean_2nd,fuy_adv_mean_2nd,fuz_adv_mean_2nd,fuz_adv_mean_prog,fuz_adv_mean_2nd_prog, &
       & fux_corr_2nd,fuy_corr_2nd,corr_dudt_2nd,                                                    &
       & fvx_adv_mean_2nd,fvy_adv_mean_2nd,fvz_adv_mean_2nd,fvz_adv_mean_prog,fvz_adv_mean_2nd_prog, &
       & fvx_corr_2nd,fvy_corr_2nd,corr_dvdt_2nd,                                                    &
       & fwx_adv_mean_2nd,fwy_adv_mean_2nd,fwz_adv_mean_2nd,fwz_adv_mean_prog,fwz_adv_mean_2nd_prog, &
       & fwx_corr_2nd,fwy_corr_2nd,corr_dwdt_2nd,                                                    &
       & ftx_sgs_mean,fty_sgs_mean,ftz_sgs_mean,                                                 &
       & fqx_sgs_mean,fqy_sgs_mean,fqz_sgs_mean,                                                    &
       & fux_sgs_mean,fuz_sgs_mean,                                                                &
       & fvx_sgs_mean,fvy_sgs_mean,fvz_sgs_mean,                                                    &
       & fwx_sgs_mean,fwy_sgs_mean,fwz_sgs_mean,                                                    &
       & t_tend_radlw_mean,t_tend_radsw_mean,t_tend_mp_mean,q_tend_mp_mean,                      &
       & u_tend_pg_mean,u_tend_cor_curv_mean,                                                    &
       & v_tend_pg_mean,v_tend_cor_curv_mean,                                                    &
       & w_tend_pg_mean,w_tend_cor_curv_mean,                                                    &
       & rhod_mean,mut_mean,z_mean,dph_x_mean,dph_y_mean                                          )

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte

    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags

    REAL, OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(OUT) ::              &
       t_mean,tx_mean,ty_mean,tz_mean,tt_mean,ftx_adv_mean,fty_adv_mean,ftz_adv_mean, &
       q_mean,qx_mean,qy_mean,qz_mean,qq_mean,fqx_adv_mean,fqy_adv_mean,fqz_adv_mean,          &
       u_mean,ux_mean,uy_mean,uz_mean,fux_adv_mean,fuy_adv_mean,fuz_adv_mean,                  &
       v_mean,vx_mean,vy_mean,vz_mean,fvx_adv_mean,fvy_adv_mean,fvz_adv_mean,                  &
       ww_mean,wd_mean,zwind_mean,wx_mean,wy_mean,wz_mean,fwx_adv_mean,fwy_adv_mean,fwz_adv_mean,  &
       ftx_corr,fty_corr,corr_dtdt,                                                           &
       fqx_corr,fqy_corr,corr_dqdt,                                                              &
       fux_corr,fuy_corr,corr_dudt,                                                              &
       fvx_corr,fvy_corr,corr_dvdt,                                                              &
       fwx_corr,fwy_corr,corr_dwdt,                                                              &
       ftx_adv_mean_2nd,fty_adv_mean_2nd,ftz_adv_mean_2nd,ftz_adv_mean_prog,ftz_adv_mean_2nd_prog, &
       ftx_corr_2nd,fty_corr_2nd,corr_dtdt_2nd,                                                    &
       fqx_adv_mean_2nd,fqy_adv_mean_2nd,fqz_adv_mean_2nd,fqz_adv_mean_prog,fqz_adv_mean_2nd_prog, &
       fqx_corr_2nd,fqy_corr_2nd,corr_dqdt_2nd,                                                    &
       fux_adv_mean_2nd,fuy_adv_mean_2nd,fuz_adv_mean_2nd,fuz_adv_mean_prog,fuz_adv_mean_2nd_prog, &
       fux_corr_2nd,fuy_corr_2nd,corr_dudt_2nd,                                                    &
       fvx_adv_mean_2nd,fvy_adv_mean_2nd,fvz_adv_mean_2nd,fvz_adv_mean_prog,fvz_adv_mean_2nd_prog, &
       fvx_corr_2nd,fvy_corr_2nd,corr_dvdt_2nd,                                                    &
       fwx_adv_mean_2nd,fwy_adv_mean_2nd,fwz_adv_mean_2nd,fwz_adv_mean_prog,fwz_adv_mean_2nd_prog, &
       fwx_corr_2nd,fwy_corr_2nd,corr_dwdt_2nd,                                                    &
       ftx_sgs_mean,fty_sgs_mean,ftz_sgs_mean,                                                 &
       fqx_sgs_mean,fqy_sgs_mean,fqz_sgs_mean,                                                    &
       fux_sgs_mean,fuz_sgs_mean,                                                                &
       fvx_sgs_mean,fvy_sgs_mean,fvz_sgs_mean,                                                    &
       fwx_sgs_mean,fwy_sgs_mean,fwz_sgs_mean,                                                    &
       t_tend_radlw_mean,t_tend_radsw_mean,t_tend_mp_mean,q_tend_mp_mean,                      &
       u_tend_pg_mean,u_tend_cor_curv_mean,                                                    &
       v_tend_pg_mean,v_tend_cor_curv_mean,                                                    &
       w_tend_pg_mean,w_tend_cor_curv_mean,                                                    &
       rhod_mean,z_mean,dph_x_mean,dph_y_mean

    REAL, OPTIONAL, DIMENSION( ims:ime , jms:jme ) , INTENT(OUT) :: mut_mean

    INTEGER :: i,j,k

    if (config_flags%do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = 0.
             avgflx_rvm(i,k,j) = 0.
             avgflx_wwm(i,k,j) = 0.
          end DO
       end DO
    end DO
    endif

    if ((config_flags%do_avgflx_cugd .EQ. 1) .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = 0.
                avgflx_cfd1(i,k,j) = 0.
                avgflx_dfu1(i,k,j) = 0.
                avgflx_efu1(i,k,j) = 0.
                avgflx_dfd1(i,k,j) = 0.
                avgflx_efd1(i,k,j) = 0.
             end DO
          end DO
       end DO
    end if

    if (present(z_mean) .and. config_flags%output_avgfluxes) then
      DO j=jts,jte
      DO i=its,ite
          mut_mean(i,j) = 0.
      ENDDO
      ENDDO

      DO j=jts,jte
         DO k=kts,kte
            DO i=its,ite
                if (config_flags%output_t_fluxes > 0) then
                    if (config_flags%output_tres_fluxes) then
                        t_mean(i,k,j) = 0.
                        tx_mean(i,k,j) = 0.
                        ty_mean(i,k,j) = 0.
                        tz_mean(i,k,j) = 0.
                        tt_mean(i,k,j) = 0.
                        ftx_adv_mean(i,k,j) = 0.
                        fty_adv_mean(i,k,j) = 0.
                        ftz_adv_mean(i,k,j) = 0.
                        ftx_corr(i,k,j) = 0.
                        fty_corr(i,k,j) = 0.
                        corr_dtdt(i,k,j) = 0.

                        IF (config_flags%output_t_fluxes_add == 1) THEN
                            !2nd order fluxes
                            ftx_adv_mean_2nd(i,k,j) = 0.
                            fty_adv_mean_2nd(i,k,j) = 0.
                            ftz_adv_mean_2nd(i,k,j) = 0.
                            ftz_adv_mean_prog(i,k,j) = 0.
                            ftz_adv_mean_2nd_prog(i,k,j) = 0.
                            ftx_corr_2nd(i,k,j) = 0.
                            fty_corr_2nd(i,k,j) = 0.
                            corr_dtdt_2nd(i,k,j) = 0.
                        ENDIF
                    endif
                    if (config_flags%output_tsgs_fluxes) then
                        ftx_sgs_mean(i,k,j) = 0.
                        fty_sgs_mean(i,k,j) = 0.
                        ftz_sgs_mean(i,k,j) = 0.
                    endif
                    if (config_flags%output_t_fluxes .eq. 1) then
                        t_tend_radlw_mean(i,k,j) = 0.
                        t_tend_radsw_mean(i,k,j) = 0.
                        t_tend_mp_mean(i,k,j) = 0.
                    endif
                endif
                if (config_flags%output_q_fluxes > 0) then
                    if (config_flags%output_qres_fluxes) then
                        q_mean(i,k,j) = 0.
                        qx_mean(i,k,j) = 0.
                        qy_mean(i,k,j) = 0.
                        qz_mean(i,k,j) = 0.
                        qq_mean(i,k,j) = 0.
                        fqx_adv_mean(i,k,j) = 0.
                        fqy_adv_mean(i,k,j) = 0.
                        fqz_adv_mean(i,k,j) = 0.
                        fqx_corr(i,k,j) = 0.
                        fqy_corr(i,k,j) = 0.
                        corr_dqdt(i,k,j) = 0.

                        IF (config_flags%output_q_fluxes_add == 1) THEN
                            !2nd order fluxes
                            fqx_adv_mean_2nd(i,k,j) = 0.
                            fqy_adv_mean_2nd(i,k,j) = 0.
                            fqz_adv_mean_2nd(i,k,j) = 0.
                            fqz_adv_mean_prog(i,k,j) = 0.
                            fqz_adv_mean_2nd_prog(i,k,j) = 0.
                            fqx_corr_2nd(i,k,j) = 0.
                            fqy_corr_2nd(i,k,j) = 0.
                            corr_dqdt_2nd(i,k,j) = 0.
                        ENDIF
                    endif
                    if (config_flags%output_qsgs_fluxes) then
                        fqx_sgs_mean(i,k,j) = 0.
                        fqy_sgs_mean(i,k,j) = 0.
                        fqz_sgs_mean(i,k,j) = 0.
                    endif
                    if (config_flags%output_q_fluxes .eq. 1) then
                        q_tend_mp_mean(i,k,j) = 0.
                    endif
                endif
                if (config_flags%output_u_fluxes > 0) then
                    if (config_flags%output_ures_fluxes) then
                        u_mean(i,k,j) = 0.
                        ux_mean(i,k,j) = 0.
                        uy_mean(i,k,j) = 0.
                        uz_mean(i,k,j) = 0.
                        fux_adv_mean(i,k,j) = 0.
                        fuy_adv_mean(i,k,j) = 0.
                        fuz_adv_mean(i,k,j) = 0.
                        fux_corr(i,k,j) = 0.
                        fuy_corr(i,k,j) = 0.
                        corr_dudt(i,k,j) = 0.

                        IF (config_flags%output_u_fluxes_add == 1) THEN
                            !2nd order fluxes
                            fux_adv_mean_2nd(i,k,j) = 0.
                            fuy_adv_mean_2nd(i,k,j) = 0.
                            fuz_adv_mean_2nd(i,k,j) = 0.
                            fuz_adv_mean_prog(i,k,j) = 0.
                            fuz_adv_mean_2nd_prog(i,k,j) = 0.
                            fux_corr_2nd(i,k,j) = 0.
                            fuy_corr_2nd(i,k,j) = 0.
                            corr_dudt_2nd(i,k,j) = 0.
                        ENDIF
                    endif
                    if (config_flags%output_usgs_fluxes) then
                        fux_sgs_mean(i,k,j) = 0.
                        fvx_sgs_mean(i,k,j) = 0.
                        fuz_sgs_mean(i,k,j) = 0.
                    endif
                    if (config_flags%output_u_fluxes .eq. 1) then
                        u_tend_pg_mean(i,k,j) = 0.
                        u_tend_cor_curv_mean(i,k,j) = 0.
                    endif
                endif
                if (config_flags%output_v_fluxes > 0) then
                    if (config_flags%output_vres_fluxes) then
                        v_mean(i,k,j) = 0.
                        vx_mean(i,k,j) = 0.
                        vy_mean(i,k,j) = 0.
                        vz_mean(i,k,j) = 0.
                        fvx_adv_mean(i,k,j) = 0.
                        fvy_adv_mean(i,k,j) = 0.
                        fvz_adv_mean(i,k,j) = 0.
                        fvx_corr(i,k,j) = 0.
                        fvy_corr(i,k,j) = 0.
                        corr_dvdt(i,k,j) = 0.

                        IF (config_flags%output_v_fluxes_add == 1) THEN
                            !2nd order fluxes
                            fvx_adv_mean_2nd(i,k,j) = 0.
                            fvy_adv_mean_2nd(i,k,j) = 0.
                            fvz_adv_mean_2nd(i,k,j) = 0.
                            fvz_adv_mean_prog(i,k,j) = 0.
                            fvz_adv_mean_2nd_prog(i,k,j) = 0.
                            fvx_corr_2nd(i,k,j) = 0.
                            fvy_corr_2nd(i,k,j) = 0.
                            corr_dvdt_2nd(i,k,j) = 0.
                        ENDIF
                    endif
                    if (config_flags%output_vsgs_fluxes) then
                        fvx_sgs_mean(i,k,j) = 0.
                        fvy_sgs_mean(i,k,j) = 0.
                        fvz_sgs_mean(i,k,j) = 0.
                    endif
                    if (config_flags%output_v_fluxes .eq. 1) then
                        v_tend_pg_mean(i,k,j) = 0.
                        v_tend_cor_curv_mean(i,k,j) = 0.
                    endif
                endif
                if (config_flags%output_w_fluxes > 0) then
                    if (config_flags%output_wres_fluxes) then
                        ww_mean(i,k,j) = 0.
                        wd_mean(i,k,j) = 0.
                        zwind_mean(i,k,j) = 0.
                        wx_mean(i,k,j) = 0.
                        wy_mean(i,k,j) = 0.
                        wz_mean(i,k,j) = 0.
                        fwx_adv_mean(i,k,j) = 0.
                        fwy_adv_mean(i,k,j) = 0.
                        fwz_adv_mean(i,k,j) = 0.
                        fwx_corr(i,k,j) = 0.
                        fwy_corr(i,k,j) = 0.
                        corr_dwdt(i,k,j) = 0.

                        IF (config_flags%output_w_fluxes_add == 1) THEN
                            !2nd order fluxes
                            fwx_adv_mean_2nd(i,k,j) = 0.
                            fwy_adv_mean_2nd(i,k,j) = 0.
                            fwz_adv_mean_2nd(i,k,j) = 0.
                            fwz_adv_mean_prog(i,k,j) = 0.
                            fwz_adv_mean_2nd_prog(i,k,j) = 0.
                            fwx_corr_2nd(i,k,j) = 0.
                            fwy_corr_2nd(i,k,j) = 0.
                            corr_dwdt_2nd(i,k,j) = 0.
                        ENDIF
                    endif
                    if (config_flags%output_wsgs_fluxes) then
                        fwx_sgs_mean(i,k,j) = 0.
                        fwy_sgs_mean(i,k,j) = 0.
                        fwz_sgs_mean(i,k,j) = 0.
                    endif
                    if (config_flags%output_w_fluxes .eq. 1) then
                        w_tend_pg_mean(i,k,j) = 0.
                        w_tend_cor_curv_mean(i,k,j) = 0.
                    endif
                endif

                rhod_mean(i,k,j) = 0.
                z_mean(i,k,j) = 0.
                dph_x_mean(i,k,j) = 0.
                dph_y_mean(i,k,j) = 0.

            end DO
         end DO
       end DO
    endif

    return
  end subroutine zero_avgflx

  subroutine upd_avgflx(avgflx_count,avgflx_rum,avgflx_rvm,avgflx_wwm, &
       &   ru_m, rv_m, ww_m, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte, do_cu,    &
       & cfu1,cfd1,dfu1,efu1,dfd1,efd1,          &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1,   &
       & config_flags,                                                              &
       & T_MEAN,TX_MEAN,TY_MEAN,TZ_MEAN,TT_MEAN,FTX_ADV_MEAN,FTY_ADV_MEAN,FTZ_ADV_MEAN, &
       & Q_MEAN,QX_MEAN,QY_MEAN,QZ_MEAN,QQ_MEAN,FQX_ADV_MEAN,FQY_ADV_MEAN,FQZ_ADV_MEAN,          &
       & U_MEAN,UX_MEAN,UY_MEAN,UZ_MEAN,FUX_ADV_MEAN,FUY_ADV_MEAN,FUZ_ADV_MEAN,                  &
       & V_MEAN,VX_MEAN,VY_MEAN,VZ_MEAN,FVX_ADV_MEAN,FVY_ADV_MEAN,FVZ_ADV_MEAN,                  &
       & WW_MEAN,WD_MEAN,ZWIND_MEAN,WX_MEAN,WY_MEAN,WZ_MEAN,FWX_ADV_MEAN,FWY_ADV_MEAN,FWZ_ADV_MEAN,  &
       & FTX_CORR_MEAN,FTY_CORR_MEAN,CORR_DTDT_MEAN,                                            &
       & FQX_CORR_MEAN,FQY_CORR_MEAN,CORR_DQDT_MEAN,                                               &
       & FUX_CORR_MEAN,FUY_CORR_MEAN,CORR_DUDT_MEAN,                                               &
       & FVX_CORR_MEAN,FVY_CORR_MEAN,CORR_DVDT_MEAN,                                               &
       & FWX_CORR_MEAN,FWY_CORR_MEAN,CORR_DWDT_MEAN,                                               &
       & FTX_ADV_MEAN_2ND,FTY_ADV_MEAN_2ND,FTZ_ADV_MEAN_2ND,FTZ_ADV_MEAN_PROG,FTZ_ADV_MEAN_2ND_PROG, &
       & FTX_CORR_MEAN_2ND,FTY_CORR_MEAN_2ND,CORR_DTDT_MEAN_2ND,                                     &
       & FQX_ADV_MEAN_2ND,FQY_ADV_MEAN_2ND,FQZ_ADV_MEAN_2ND,FQZ_ADV_MEAN_PROG,FQZ_ADV_MEAN_2ND_PROG, &
       & FQX_CORR_MEAN_2ND,FQY_CORR_MEAN_2ND,CORR_DQDT_MEAN_2ND,                                     &
       & FUX_ADV_MEAN_2ND,FUY_ADV_MEAN_2ND,FUZ_ADV_MEAN_2ND,FUZ_ADV_MEAN_PROG,FUZ_ADV_MEAN_2ND_PROG, &
       & FUX_CORR_MEAN_2ND,FUY_CORR_MEAN_2ND,CORR_DUDT_MEAN_2ND,                                     &
       & FVX_ADV_MEAN_2ND,FVY_ADV_MEAN_2ND,FVZ_ADV_MEAN_2ND,FVZ_ADV_MEAN_PROG,FVZ_ADV_MEAN_2ND_PROG, &
       & FVX_CORR_MEAN_2ND,FVY_CORR_MEAN_2ND,CORR_DVDT_MEAN_2ND,                                     &
       & FWX_ADV_MEAN_2ND,FWY_ADV_MEAN_2ND,FWZ_ADV_MEAN_2ND,FWZ_ADV_MEAN_PROG,FWZ_ADV_MEAN_2ND_PROG, &
       & FWX_CORR_MEAN_2ND,FWY_CORR_MEAN_2ND,CORR_DWDT_MEAN_2ND,                                     &
       & FTX_SGS_MEAN,FTY_SGS_MEAN,FTZ_SGS_MEAN,                                                 &
       & FQX_SGS_MEAN,FQY_SGS_MEAN,FQZ_SGS_MEAN,                                                    &
       & FUX_SGS_MEAN,FUZ_SGS_MEAN,                                                                &
       & FVX_SGS_MEAN,FVY_SGS_MEAN,FVZ_SGS_MEAN,                                                    &
       & FWX_SGS_MEAN,FWY_SGS_MEAN,FWZ_SGS_MEAN,                                                    &
       & T_TEND_RADLW_MEAN,T_TEND_RADSW_MEAN,T_TEND_MP_MEAN,Q_TEND_MP_MEAN,                      &
       & U_TEND_PG_MEAN,U_TEND_COR_CURV_MEAN,                                                    &
       & V_TEND_PG_MEAN,V_TEND_COR_CURV_MEAN,                                                    &
       & W_TEND_PG_MEAN,W_TEND_COR_CURV_MEAN,                                                    &
       & RHOD_MEAN,MUT_MEAN,Z_MEAN,DPH_X_MEAN,DPH_Y_MEAN,w_diag,                                 &
       & ftx_adv, fty_adv, ftz_adv, ftx_small, fty_small, ftz_small, fqx_adv, fqy_adv, fqz_adv,        &
       & fux_adv, fuy_adv, fuz_adv, fvx_adv, fvy_adv, fvz_adv, fwx_adv, fwy_adv, fwz_adv,        &
       & ftx_sgs, fty_sgs, ftz_sgs, fqx_sgs, fqy_sgs, fqz_sgs,                                      &
       & fux_sgs, fuz_sgs, fvx_sgs, fvy_sgs, fvz_sgs, fwx_sgs, fwy_sgs, fwz_sgs,                         &
       & t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp,                                      &
       & u_tend_pg,u_tend_cor_curv,                                                              &
       & v_tend_pg,v_tend_cor_curv,                                                              &
       & w_tend_pg,w_tend_cor_curv,                                                              &
       & U, u_save, V, v_save, W, w_save, ru, rv, ww, thm, Q,                                                            &
       & ph_1, ph_2, phb, zx, zy, dph_x, dph_y,                                                  &
       & dnw, fnm, fnp, mut, c1h, c2h, c1f, c2f, cf1, cf2, cf3, cfn, cfn1,                       &
       & msfuy, msfvx, msfvy, msftx, msfty,                                                      &
       & last_timestep, dt                                                                       )

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte

    INTEGER , INTENT(IN)        :: avgflx_count
    LOGICAL, INTENT(IN) :: do_cu
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: ru_m, &
         rv_m, &
         ww_m

    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(IN) ::    &
         cfu1,cfd1,dfu1,efu1,dfd1,efd1
    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1


    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::            &
       T_MEAN,TX_MEAN,TY_MEAN,TZ_MEAN,TT_MEAN,FTX_ADV_MEAN,FTY_ADV_MEAN,FTZ_ADV_MEAN, &
       Q_MEAN,QX_MEAN,QY_MEAN,QZ_MEAN,QQ_MEAN,FQX_ADV_MEAN,FQY_ADV_MEAN,FQZ_ADV_MEAN,          &
       U_MEAN,UX_MEAN,UY_MEAN,UZ_MEAN,FUX_ADV_MEAN,FUY_ADV_MEAN,FUZ_ADV_MEAN,                  &
       V_MEAN,VX_MEAN,VY_MEAN,VZ_MEAN,FVX_ADV_MEAN,FVY_ADV_MEAN,FVZ_ADV_MEAN,                  &
       WW_MEAN,WD_MEAN,ZWIND_MEAN,WX_MEAN,WY_MEAN,WZ_MEAN,FWX_ADV_MEAN,FWY_ADV_MEAN,FWZ_ADV_MEAN,  &
       FTX_CORR_MEAN,FTY_CORR_MEAN,CORR_DTDT_MEAN,                                                           &
       FQX_CORR_MEAN,FQY_CORR_MEAN,CORR_DQDT_MEAN,                                                              &
       FUX_CORR_MEAN,FUY_CORR_MEAN,CORR_DUDT_MEAN,                                                              &
       FVX_CORR_MEAN,FVY_CORR_MEAN,CORR_DVDT_MEAN,                                                              &
       FWX_CORR_MEAN,FWY_CORR_MEAN,CORR_DWDT_MEAN,                                                              &
       FTX_ADV_MEAN_2ND,FTY_ADV_MEAN_2ND,FTZ_ADV_MEAN_2ND,FTZ_ADV_MEAN_PROG,FTZ_ADV_MEAN_2ND_PROG, &
       FTX_CORR_MEAN_2ND,FTY_CORR_MEAN_2ND,CORR_DTDT_MEAN_2ND,                                     &
       FQX_ADV_MEAN_2ND,FQY_ADV_MEAN_2ND,FQZ_ADV_MEAN_2ND,FQZ_ADV_MEAN_PROG,FQZ_ADV_MEAN_2ND_PROG, &
       FQX_CORR_MEAN_2ND,FQY_CORR_MEAN_2ND,CORR_DQDT_MEAN_2ND,                                     &
       FUX_ADV_MEAN_2ND,FUY_ADV_MEAN_2ND,FUZ_ADV_MEAN_2ND,FUZ_ADV_MEAN_PROG,FUZ_ADV_MEAN_2ND_PROG, &
       FUX_CORR_MEAN_2ND,FUY_CORR_MEAN_2ND,CORR_DUDT_MEAN_2ND,                                     &
       FVX_ADV_MEAN_2ND,FVY_ADV_MEAN_2ND,FVZ_ADV_MEAN_2ND,FVZ_ADV_MEAN_PROG,FVZ_ADV_MEAN_2ND_PROG, &
       FVX_CORR_MEAN_2ND,FVY_CORR_MEAN_2ND,CORR_DVDT_MEAN_2ND,                                     &
       FWX_ADV_MEAN_2ND,FWY_ADV_MEAN_2ND,FWZ_ADV_MEAN_2ND,FWZ_ADV_MEAN_PROG,FWZ_ADV_MEAN_2ND_PROG, &
       FWX_CORR_MEAN_2ND,FWY_CORR_MEAN_2ND,CORR_DWDT_MEAN_2ND,                                     &
       FTX_SGS_MEAN,FTY_SGS_MEAN,FTZ_SGS_MEAN,                                                 &
       FQX_SGS_MEAN,FQY_SGS_MEAN,FQZ_SGS_MEAN,                                                    &
       FUX_SGS_MEAN,FUZ_SGS_MEAN,                                                                &
       FVX_SGS_MEAN,FVY_SGS_MEAN,FVZ_SGS_MEAN,                                                    &
       FWX_SGS_MEAN,FWY_SGS_MEAN,FWZ_SGS_MEAN,                                                    &
       T_TEND_RADLW_MEAN,T_TEND_RADSW_MEAN,T_TEND_MP_MEAN,Q_TEND_MP_MEAN,                      &
       U_TEND_PG_MEAN,U_TEND_COR_CURV_MEAN,                                                    &
       V_TEND_PG_MEAN,V_TEND_COR_CURV_MEAN,                                                    &
       W_TEND_PG_MEAN,W_TEND_COR_CURV_MEAN,                                                    &
       RHOD_MEAN,Z_MEAN,DPH_X_MEAN,DPH_Y_MEAN

    REAL, OPTIONAL, DIMENSION( ims:ime , jms:jme ) , INTENT(OUT) :: MUT_MEAN

    REAL, OPTIONAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(OUT) :: w_diag

    REAL, OPTIONAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
         U, u_save, V, v_save, W, w_save, ru, rv, ww,  &
         thm, Q, ph_1, ph_2, phb, zx, zy,  &
         ftx_adv, fty_adv, ftz_adv,     &
         fqx_adv, fqy_adv, fqz_adv,        &
         fux_adv, fuy_adv, fuz_adv,        &
         fvx_adv, fvy_adv, fvz_adv,        &
         fwx_adv, fwy_adv, fwz_adv,        &
         t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp, &
         u_tend_pg,u_tend_cor_curv,                         &
         v_tend_pg,v_tend_cor_curv,                         &
         w_tend_pg,w_tend_cor_curv,                         &
         ftx_sgs, fty_sgs, ftz_sgs, fqx_sgs, fqy_sgs, fqz_sgs, &
         fux_sgs, fuz_sgs, fvx_sgs, fvy_sgs, fvz_sgs, fwx_sgs, fwy_sgs, fwz_sgs

    REAL, OPTIONAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(INOUT) :: ftx_small, fty_small, ftz_small, &
                                                                            dph_x, dph_y

    REAL , DIMENSION( kms:kme ) , OPTIONAL, INTENT(IN   ) :: dnw, fnm, fnp, c1h, c2h, c1f, c2f
    REAL , DIMENSION( ims:ime, jms:jme ), OPTIONAL, INTENT(IN   ) :: mut
    REAL, OPTIONAL, INTENT(IN) :: dt, cf1, cf2, cf3, cfn, cfn1
    REAL, OPTIONAL, DIMENSION( ims:ime , jms:jme ) ,  INTENT(IN   ) :: msfuy,  &
                                                                       msfvx,  &
                                                                       msfvy,  &
                                                                       msftx,  &
                                                                       msfty

    LOGICAL, OPTIONAL, INTENT(IN) :: last_timestep


    REAL,    DIMENSION(ims:ime, kms:kme, jms:jme) ::    rho_d, th
    REAL,    DIMENSION( its-1:ite+1 , kts:kte , jts-1:jte+1 ) ::    thd, dzdt, u8w, v8w, rho8w, rw_diag, zxu, zyv

    INTEGER :: i,j,k
    REAL :: local_count, rhom, rhox, rhoy, rhoz, rhoxy, rhoxz, rhoyz,                      &
            rho_mean, rhox_mean, rhoy_mean, rhoz_mean, rhoxy_mean, rhoxz_mean, rhoyz_mean, &
            rho_mean_flx, rhox_mean_flx, rhoy_mean_flx, rhoz_mean_flx, rhoxy_mean_flx,     &
            rhoxz_mean_flx, rhoyz_mean_flx, &
            mu_mean, mux_mean, muy_mean, muz_mean, muxy_mean, muxz_mean, muyz_mean,        &
            fqx_dfac, fqy_dfac, fqz_dfac, &
            fux_dfac, fuy_dfac, &
            fvx_dfac, fvy_dfac, &
            fwx_dfac, fwy_dfac, &
            thetax, thetay, thetaz, qx, qy, qz, q_stag, ux, uy, uz, vx, vy, vz, wx, wy, wz,     &
            ru_d, rv_d, ww_d,               &
            rho_s, rzxu, rzyv, rdzdt, &
            mu, dphidn

    REAL :: thetax2, thetay2, thetaz2,                 &
            qx2, qy2, qz2,                             &
            ux2, uy2, uz2, rhoxz_flx,                  &
            vx2, vy2, vz2, rhoyz_flx,                  &
            wx2, wy2, wz2,                             &
            rw_diag_s, w_s


    local_count = real(avgflx_count)

    if (present(dt) .and. config_flags%output_avgfluxes) then

    DO j=jts-1,jte
    DO k=kts,min(kte,kde-1)
    DO i=its-1,ite
      !calculate dry density
      mu = c1h(k)*mut(i,j)+c2h(k)
      dphidn = ((phb(i,k+1,j)+ph_2(i,k+1,j)) - (phb(i,k,j)+ph_2(i,k,j)))/dnw(k)
      rho_d(i,k,j) = -mu/dphidn
      RHOD_MEAN(i,k,j) = (local_count*RHOD_MEAN(i,k,j)  + rho_d(i,k,j)  )/(local_count+1.)
      IF (config_flags%output_tres_fluxes) THEN
          IF ( config_flags%use_theta_m .EQ. 1 ) THEN
            thd(i,k,j) = (thm(i,k,j) + t0)/(1 + rvovrd*Q(i,k,j)) - t0
            IF ( config_flags%output_dry_theta_fluxes) THEN
              th(i,k,j) = thd(i,k,j)
            ELSE
              th(i,k,j) = thm(i,k,j)
            ENDIF
          ELSE
            thd(i,k,j) = thm(i,k,j)
            th(i,k,j) = thm(i,k,j)
          ENDIF
      ENDIF
    ENDDO
    ENDDO
    ENDDO

    k = kde
    DO j=jts-1,jte
    DO i=its-1,ite
      !set upper boundary to prevent errors
      rho_d(i,k,j) = rho_d(i,k-1,j)
      RHOD_MEAN(i,k,j) = RHOD_MEAN(i,k-1,j)
      IF (config_flags%output_tres_fluxes) THEN
        thd(i,k,j) = thd(i,k-1,j)
        th(i,k,j) = th(i,k-1,j)
      ENDIF
    ENDDO
    ENDDO

    DO j=jts-1,jte
    DO i=its-1  ,ite
        MUT_MEAN(i,j) = (local_count*MUT_MEAN(i,j) + mut(i,j))/(local_count+1.)
    ENDDO
    ENDDO

    IF (config_flags%output_res_fluxes) THEN
        DO j=jts-1,jte+1
        DO k=kts,kte
        DO i=its-1,ite+1
            dzdt(i,k,j) = (ph_2(i,k,j) - ph_1(i,k,j))/(g*dt)
            call stagger_z(u8w(i,k,j), u_save(i,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           k, kms, kme, kde              )
            call stagger_z(v8w(i,k,j), v_save(i,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           k, kms, kme, kde              )
            call stagger_z(rho8w(i,k,j), rho_d(i,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           k, kms, kme, kde              )
        ENDDO
        ENDDO
        ENDDO
        DO j=jts-1,jte
        DO k=kts,kte
        DO i=its-1,ite
            !diagnostic velocity
            IF (k .eq. kts) THEN
                dph_x(i,k,j) = -g*0.5*(zx(i,k,j)*u8w(i,k,j) + zx(i+1,k,j)*u8w(i+1,k,j))
                dph_y(i,k,j) = -g*0.5*(zy(i,k,j)*v8w(i,k,j) + zy(i,k,j+1)*v8w(i,k,j+1))
            ENDIF
            zxu(i,k,j) = -dph_x(i,k,j)/g
            zyv(i,k,j) = -dph_y(i,k,j)/g
            w_diag(i,k,j) = dzdt(i,k,j) + zxu(i,k,j) + zyv(i,k,j) + ww(i,k,j)*msfty(i,j)/(-rho8w(i,k,j)*g)
            rw_diag(i,k,j) = w_diag(i,k,j)*rho8w(i,k,j)
        ENDDO
        ENDDO
        ENDDO
    ENDIF

    DO j=jts,jte
      DO k=kts,kte
        DO i=its,ite
          Z_MEAN(i,k,j) = (local_count*Z_MEAN(i,k,j)  + (phb(i,k,j)+ph_2(i,k,j))/g  )/(local_count+1.)
          rho_mean   = 1.
          rhox_mean  = 1.
          rhoy_mean  = 1.
          rhoxy_mean = 1.
          rhoz_mean  = 1.
          rhoxz_mean = 1.
          rhoyz_mean = 1.
          rho_mean_flx = 1.
          rhox_mean_flx  = 1.
          rhoy_mean_flx  = 1.
          rhoxy_mean_flx = 1.
          rhoz_mean_flx  = 1.
          rhoxz_mean_flx = 1.
          rhoyz_mean_flx = 1.

          fqx_dfac = 1.
          fqy_dfac = 1.
          fqz_dfac = 1.
          fux_dfac = 1.
          fuy_dfac = 1.
          fvx_dfac = 1.
          fvy_dfac = 1.
          fwx_dfac = 1.
          fwy_dfac = 1.
          IF (last_timestep) THEN
              !mean staggered densities to normalize fluxes
              rho_mean_flx = RHOD_MEAN(i,k,j)
              call stagger_z(rhoz_mean_flx, RHOD_MEAN(i,:,j), &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                           k, kms, kme, kde                   )
              call stagger_z(rhoxz_mean_flx, 0.5*(RHOD_MEAN(i,:,j) + RHOD_MEAN(i-1,:,j)), &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                           k, kms, kme, kde                   )
              call stagger_z(rhoyz_mean_flx, 0.5*(RHOD_MEAN(i,:,j) + RHOD_MEAN(i,:,j-1)), &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                           k, kms, kme, kde                   )
              rhox_mean_flx = 0.5*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i-1,k,j))
              rhoy_mean_flx = 0.5*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i,k,j-1))
              rhoxy_mean_flx = 0.25*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i,k,j-1) + RHOD_MEAN(i-1,k,j) + RHOD_MEAN(i-1,k,j-1))

              !factors to decouple horizontal fluxes from mu and map-scale factors
              muxz_mean = c1f(k)*0.5*(MUT_MEAN(i,j) + MUT_MEAN(i-1,j)) + c2f(k)
              muyz_mean = c1f(k)*0.5*(MUT_MEAN(i,j) + MUT_MEAN(i,j-1)) + c2f(k)
              fwx_dfac  = msfuy(i,j)/muxz_mean
              fwy_dfac  = msfvx(i,j)/muyz_mean
              fqz_dfac  = msfty(i,j)/(-g)
              IF (k .lt. kde) THEN
                  !k=kde only relevant for vertically staggered variables
                  mu_mean = c1h(k)*MUT_MEAN(i,j) + c2h(k)
                  mux_mean = c1h(k)*0.5*(MUT_MEAN(i,j) + MUT_MEAN(i-1,j)) + c2h(k)
                  muy_mean = c1h(k)*0.5*(MUT_MEAN(i,j) + MUT_MEAN(i,j-1)) + c2h(k)
                  muxy_mean = c1h(k)*0.25*(MUT_MEAN(i,j) + MUT_MEAN(i-1,j) + MUT_MEAN(i,j-1) + MUT_MEAN(i-1,j-1)) + c2h(k)
                  fqx_dfac  = msfuy(i,j)/mux_mean
                  fqy_dfac  = msfvx(i,j)/muy_mean
                  fux_dfac  = 0.5*(msfuy(i,j) + msfuy(i+1,j))/mu_mean
                  fuy_dfac  = 0.5*(msfvx(i,j) + msfvx(i-1,j))/muxy_mean
                  fvx_dfac  = 0.5*(msfuy(i,j) + msfuy(i,j-1))/muxy_mean
                  fvy_dfac  = 0.5*(msfvx(i,j) + msfvx(i,j+1))/mu_mean
              ENDIF
              IF (config_flags%hesselberg_avg) THEN
                !mean staggered densities to normalize means by density
                rho_mean = RHOD_MEAN(i,k,j)
                rhoz_mean = rhoz_mean_flx
                rhoxz_mean = rhoxz_mean_flx
                rhoyz_mean = rhoyz_mean_flx
                rhox_mean = rhox_mean_flx
                rhoy_mean = rhoy_mean_flx
                rhoxy_mean = rhoxy_mean_flx
              ENDIF

          ENDIF
          IF (config_flags%output_res_fluxes) THEN
            !stagger density
            if (config_flags%hesselberg_avg) then
              rhom = rho_d(i,k,j)
              rhox = 0.5*(rho_d(i,k,j)+rho_d(i-1,k,j))
              rhoy = 0.5*(rho_d(i,k,j)+rho_d(i,k,j-1))
              rhoxy = 0.25*(rho_d(i,k,j)+rho_d(i,k,j-1)+rho_d(i-1,k,j)+rho_d(i-1,k,j-1))
              rhoz = rho8w(i,k,j)
              rhoxz = 0.5*(rho8w(i,k,j)+rho8w(i-1,k,j))
              rhoyz = 0.5*(rho8w(i,k,j)+rho8w(i,k,j-1))
            else
              rhox = 1
              rhoy = 1
              rhoxy = 1
              rhoz = 1
              rhoxz = 1
              rhoyz = 1
            endif

            rho_s = rho8w(i,k,j)
            !determine staggered variables from total fluxes
            IF (config_flags%output_tres_fluxes) THEN
                call get_staggered_from_fluxes(thetax, thetay, thetaz,      &
                                               ftx_adv, fty_adv, ftz_adv,&
                                               ru(i,k,j), rv(i,k,j), ww(i,k,j), th, &
                                               cf1, cf2, cf3, cfn, cfn1, .false.,   &
                                               i, j, k,                             &
                                               ids, ide, jds, jde, kds, kde,        &
                                               ims, ime, jms, jme, kms, kme         )

                !for cartesian corrections
                rzxu = rho_s*zxu(i,k,j)
                rzyv = rho_s*zyv(i,k,j)
                rdzdt = rho_s*dzdt(i,k,j)

                !build averages
                T_MEAN(i,k,j) = (local_count*T_MEAN(i,k,j) + thd(i,k,j)*rhom )/(local_count+1.)/rho_mean
                TX_MEAN(i,k,j) = (local_count*TX_MEAN(i,k,j) + thetax*rhox )/(local_count+1.)/rhox_mean
                TY_MEAN(i,k,j) = (local_count*TY_MEAN(i,k,j) + thetay*rhoy )/(local_count+1.)/rhoy_mean
                TZ_MEAN(i,k,j) = (local_count*TZ_MEAN(i,k,j) + thetaz*rhoz )/(local_count+1.)/rhoz_mean
                TT_MEAN(i,k,j) = (local_count*TT_MEAN(i,k,j) + thd(i,k,j)**2 )/(local_count+1.)
                IF (( config_flags%use_theta_m .EQ. 1 ) .and. ( config_flags%output_dry_theta_fluxes)) THEN
                    ftx_small(i,k,j) = ftx_small(i,k,j)/(1 + rvovrd*0.5*(Q(i,k,j) + Q(i-1,k,j)))
                    fty_small(i,k,j) = fty_small(i,k,j)/(1 + rvovrd*0.5*(Q(i,k,j) + Q(i,k,j-1)))
                    call stagger_z(q_stag, Q(i,:,j), cf1, cf2, cf3, cfn, cfn1, fnm, fnp, k, kms, kme, kde   )
                    ftz_small(i,k,j) = ftz_small(i,k,j)/(1 + rvovrd*q_stag)
                ENDIF
                FTX_ADV_MEAN(i,k,j)  = (local_count*FTX_ADV_MEAN(i,k,j) + ftx_adv(i,k,j) + ftx_small(i,k,j) )/(local_count+1.)*fqx_dfac
                FTY_ADV_MEAN(i,k,j)  = (local_count*FTY_ADV_MEAN(i,k,j) + fty_adv(i,k,j) + fty_small(i,k,j) )/(local_count+1.)*fqy_dfac
                FTZ_ADV_MEAN(i,k,j)  = (local_count*FTZ_ADV_MEAN(i,k,j) + ftz_adv(i,k,j) + ftz_small(i,k,j) )/(local_count+1.)*fqz_dfac !TODOm: how to account for small in mean fluxes?
                FTX_CORR_MEAN(i,k,j)  = (local_count*FTX_CORR_MEAN(i,k,j) + rzxu*thetaz )/(local_count+1.)
                FTY_CORR_MEAN(i,k,j)  = (local_count*FTY_CORR_MEAN(i,k,j) + rzyv*thetaz )/(local_count+1.)
                CORR_DTDT_MEAN(i,k,j)  = (local_count*CORR_DTDT_MEAN(i,k,j) + rdzdt*thetaz )/(local_count+1.)
                IF (last_timestep) THEN
                  !add corrections to vertical flux
                  FTZ_ADV_MEAN(i,k,j) = (FTZ_ADV_MEAN(i,k,j) + FTX_CORR_MEAN(i,k,j) + FTY_CORR_MEAN(i,k,j) + &
                                         CORR_DTDT_MEAN(i,k,j) )/rhoz_mean_flx
                ENDIF

                IF (config_flags%output_t_fluxes_add == 1) THEN
                    !2nd order and prognostic w fluxes
                    thetax2 = 0.5*(th(i-1,k,j) + th(i,k,j))
                    thetay2 = 0.5*(th(i,k,j-1) + th(i,k,j))
                    IF (( k .gt. kts) .and. (k .lt. kte)) THEN
                      thetaz2 = (fnp(k)* th(i,k-1,j) + fnm(k)* th(i,k,j))
                    ELSE
                      thetaz2 = thetaz
                    ENDIF
                    FTX_ADV_MEAN_2ND(i,k,j) = (local_count*FTX_ADV_MEAN_2ND(i,k,j) + ( ru(i,k,j)*thetax2 ) )/(local_count+1.)*fqx_dfac
                    FTY_ADV_MEAN_2ND(i,k,j) = (local_count*FTY_ADV_MEAN_2ND(i,k,j) + ( rv(i,k,j)*thetay2 ) )/(local_count+1.)*fqy_dfac
                    FTZ_ADV_MEAN_2ND_PROG(i,k,j) = (local_count*FTZ_ADV_MEAN_2ND_PROG(i,k,j) + ( w_save(i,k,j)*thetaz2*rho8w(i,k,j) ) )/(local_count+1.)/rhoz_mean_flx
                    FTZ_ADV_MEAN_PROG(i,k,j) = (local_count*FTZ_ADV_MEAN_PROG(i,k,j) + ( w_save(i,k,j)*thetaz*rho8w(i,k,j)  ) )/(local_count+1.)/rhoz_mean_flx
                    FTZ_ADV_MEAN_2ND(i,k,j) = (local_count*FTZ_ADV_MEAN_2ND(i,k,j) + ( rw_diag(i,k,j)*thetaz2 ) )/(local_count+1.)/rhoz_mean_flx
                    FTX_CORR_MEAN_2ND(i,k,j) = (local_count*FTX_CORR_MEAN_2ND(i,k,j) + rzxu*thetaz2 )/(local_count+1.)
                    FTY_CORR_MEAN_2ND(i,k,j) = (local_count*FTY_CORR_MEAN_2ND(i,k,j) + rzyv*thetaz2 )/(local_count+1.)
                    CORR_DTDT_MEAN_2ND(i,k,j) = (local_count*CORR_DTDT_MEAN_2ND(i,k,j) + rdzdt*thetaz2 )/(local_count+1.)
                ENDIF
            ENDIF

            IF (config_flags%output_qres_fluxes) THEN
                !staggered q, different ru,rv,ww needed
                call get_staggered_from_fluxes(qx, qy, qz,                 &
                                               fqx_adv, fqy_adv, fqz_adv,  &
                                               ru_m(i,k,j), rv_m(i,k,j), ww_m(i,k,j), Q, &
                                                cf1, cf2, cf3, cfn, cfn1, .false.,       &
                                               i, j, k,                                  &
                                               ids, ide, jds, jde, kds, kde,             &
                                               ims, ime, jms, jme, kms, kme              )

                !for cartesian corrections
                rzxu = rho_s*zxu(i,k,j)
                rzyv = rho_s*zyv(i,k,j)
                rdzdt = rho_s*dzdt(i,k,j)
                !use ww_m instead of ww
                rw_diag_s = rho8w(i,k,j)*(dzdt(i,k,j) + zxu(i,k,j) + zyv(i,k,j)) + ww_m(i,k,j)*msfty(i,j)/(-g)
                !build averages
                Q_MEAN(i,k,j) = (local_count*Q_MEAN(i,k,j) + Q(i,k,j)*rhom )/(local_count+1.)/rho_mean
                QX_MEAN(i,k,j) = (local_count*QX_MEAN(i,k,j) + qx*rhox )/(local_count+1.)/rhox_mean
                QY_MEAN(i,k,j) = (local_count*QY_MEAN(i,k,j) + qy*rhoy )/(local_count+1.)/rhoy_mean
                QZ_MEAN(i,k,j) = (local_count*QZ_MEAN(i,k,j) + qz*rhoz )/(local_count+1.)/rhoz_mean
                QQ_MEAN(i,k,j) = (local_count*QQ_MEAN(i,k,j) + Q(i,k,j)*2 )/(local_count+1.)
                FQX_ADV_MEAN(i,k,j)  = (local_count*FQX_ADV_MEAN(i,k,j) + fqx_adv(i,k,j) )/(local_count+1.)*fqx_dfac
                FQY_ADV_MEAN(i,k,j)  = (local_count*FQY_ADV_MEAN(i,k,j) + fqy_adv(i,k,j) )/(local_count+1.)*fqy_dfac
                FQZ_ADV_MEAN(i,k,j)  = (local_count*FQZ_ADV_MEAN(i,k,j) + rw_diag_s*qz )/(local_count+1.)/rhoz_mean_flx
                FQX_CORR_MEAN(i,k,j)  = (local_count*FQX_CORR_MEAN(i,k,j) + rzxu*qz )/(local_count+1.)
                FQY_CORR_MEAN(i,k,j)  = (local_count*FQY_CORR_MEAN(i,k,j) + rzyv*qz )/(local_count+1.)
                CORR_DQDT_MEAN(i,k,j)  = (local_count*CORR_DQDT_MEAN(i,k,j) + rdzdt*qz )/(local_count+1.)

                IF (config_flags%output_q_fluxes_add == 1) THEN
                    !2nd order and prognostic w fluxes
                    qx2 = 0.5*(Q(i-1,k,j) + Q(i,k,j))
                    qy2 = 0.5*(Q(i,k,j-1) + Q(i,k,j))
                    IF (( k .gt. kts) .and. (k .lt. kte)) THEN
                      qz2 = (fnp(k)* Q(i,k-1,j) + fnm(k)* Q(i,k,j))
                    ELSE
                      qz2 = qz
                    ENDIF

                    FQX_ADV_MEAN_2ND(i,k,j) = (local_count*FQX_ADV_MEAN_2ND(i,k,j) + ( ru_m(i,k,j)*qx2 ) )/(local_count+1.)*fqx_dfac
                    FQY_ADV_MEAN_2ND(i,k,j) = (local_count*FQY_ADV_MEAN_2ND(i,k,j) + ( rv_m(i,k,j)*qy2 ) )/(local_count+1.)*fqy_dfac
                    FQZ_ADV_MEAN_2ND_PROG(i,k,j) = (local_count*FQZ_ADV_MEAN_2ND_PROG(i,k,j) + ( w_save(i,k,j)*qz2*rho8w(i,k,j) ) )/(local_count+1.)/rhoz_mean_flx
                    FQZ_ADV_MEAN_PROG(i,k,j) = (local_count*FQZ_ADV_MEAN_PROG(i,k,j) + ( w_save(i,k,j)*qz*rho8w(i,k,j)  ) )/(local_count+1.)/rhoz_mean_flx
                    FQZ_ADV_MEAN_2ND(i,k,j) = (local_count*FQZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*qz2 )/(local_count+1.)/rhoz_mean_flx
                    FQX_CORR_MEAN_2ND(i,k,j) = (local_count*FQX_CORR_MEAN_2ND(i,k,j) + rzxu*qz2 )/(local_count+1.)
                    FQY_CORR_MEAN_2ND(i,k,j) = (local_count*FQY_CORR_MEAN_2ND(i,k,j) + rzyv*qz2 )/(local_count+1.)
                    CORR_DQDT_MEAN_2ND(i,k,j) = (local_count*CORR_DQDT_MEAN_2ND(i,k,j) + rdzdt*qz2 )/(local_count+1.)
                ENDIF
            ENDIF

            IF (config_flags%output_ures_fluxes) THEN
                !(de)stagger in x direction
                ru_d = 0.5*(ru(i,k,j)+ru(i+1,k,j))
                rv_d = 0.5*(rv(i-1,k,j)+rv(i,k,j))
                ww_d = 0.5*(ww(i-1,k,j)+ww(i,k,j))
                call get_staggered_from_fluxes(ux, uy, uz,                 &
                                               fux_adv, fuy_adv, fuz_adv,  &
                                               ru_d, rv_d, ww_d, u_save,        &
                                               cf1, cf2, cf3, cfn, cfn1, .false., &
                                               i, j, k,                           &
                                               ids, ide, jds, jde, kds, kde,      &
                                               ims, ime, jms, jme, kms, kme       )

                rho_s = 0.5*(rho8w(i,k,j) + rho8w(i-1,k,j))
                !for cartesian corrections
                rzxu = rho_s*0.5*(zxu(i-1,k,j)+zxu(i,k,j))
                rzyv = rho_s*0.5*(zyv(i-1,k,j)+zyv(i,k,j))
                rdzdt = rho_s*0.5*(dzdt(i-1,k,j)+dzdt(i,k,j))
                rw_diag_s = 0.5*(rw_diag(i,k,j) + rw_diag(i-1,k,j))

                !build averages
                UX_MEAN(i,k,j) = (local_count*UX_MEAN(i,k,j) + ux*rhom  )/(local_count+1.)/rho_mean
                UY_MEAN(i,k,j) = (local_count*UY_MEAN(i,k,j) + uy*rhoxy )/(local_count+1.)/rhoxy_mean
                UZ_MEAN(i,k,j) = (local_count*UZ_MEAN(i,k,j) + uz*rhoxz )/(local_count+1.)/rhoxz_mean
                FUX_ADV_MEAN(i,k,j)  = (local_count*FUX_ADV_MEAN(i,k,j) + fux_adv(i,k,j) )/(local_count+1.)*fux_dfac
                FUY_ADV_MEAN(i,k,j)  = (local_count*FUY_ADV_MEAN(i,k,j) + fuy_adv(i,k,j) )/(local_count+1.)*fuy_dfac
                FUZ_ADV_MEAN(i,k,j)  = (local_count*FUZ_ADV_MEAN(i,k,j) + rw_diag_s*uz )/(local_count+1.)/rhoxz_mean_flx
                FUX_CORR_MEAN(i,k,j)  = (local_count*FUX_CORR_MEAN(i,k,j) + rzxu*uz )/(local_count+1.)
                FUY_CORR_MEAN(i,k,j)  = (local_count*FUY_CORR_MEAN(i,k,j) + rzyv*uz )/(local_count+1.)
                CORR_DUDT_MEAN(i,k,j)  = (local_count*CORR_DUDT_MEAN(i,k,j) + rdzdt*uz )/(local_count+1.)

                IF (config_flags%output_u_fluxes_add == 1) THEN
                    !2nd order and prognostic w fluxes
                    ux2 = 0.5*(u_save(i+1,k,j) + u_save(i,k,j))
                    uy2 = 0.5*(u_save(i,k,j-1) + u_save(i,k,j))
                    IF (( k .gt. kts) .and. (k .lt. kte)) THEN
                      uz2 = (fnp(k)* u_save(i,k-1,j) + fnm(k)* u_save(i,k,j))
                    ELSE
                      uz2 = uz
                    ENDIF
                    w_s = 0.5*(w_save(i-1,k,j)+w_save(i,k,j))
                    FUX_ADV_MEAN_2ND(i,k,j) = (local_count*FUX_ADV_MEAN_2ND(i,k,j) + ru_d*ux2 )/(local_count+1.)*fux_dfac
                    FUY_ADV_MEAN_2ND(i,k,j) = (local_count*FUY_ADV_MEAN_2ND(i,k,j) + rv_d*uy2 )/(local_count+1.)*fuy_dfac
                    FUZ_ADV_MEAN_2ND_PROG(i,k,j) = (local_count*FUZ_ADV_MEAN_2ND_PROG(i,k,j) + w_s*uz2*rho_s )/(local_count+1.)/rhoxz_mean_flx
                    FUZ_ADV_MEAN_PROG(i,k,j) = (local_count*FUZ_ADV_MEAN_PROG(i,k,j) + w_s*uz*rho_s )/(local_count+1.)/rhoxz_mean_flx
                    FUZ_ADV_MEAN_2ND(i,k,j) = (local_count*FUZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*uz2 )/(local_count+1.)/rhoxz_mean_flx
                    FUX_CORR_MEAN_2ND(i,k,j) = (local_count*FUX_CORR_MEAN_2ND(i,k,j) + rzxu*uz2 )/(local_count+1.)
                    FUY_CORR_MEAN_2ND(i,k,j) = (local_count*FUY_CORR_MEAN_2ND(i,k,j) + rzyv*uz2 )/(local_count+1.)
                    CORR_DUDT_MEAN_2ND(i,k,j) = (local_count*CORR_DUDT_MEAN_2ND(i,k,j) + rdzdt*uz2 )/(local_count+1.)
                ENDIF
            ENDIF

            IF (config_flags%output_vres_fluxes) THEN
                !(de)stagger in y direction
                ru_d = 0.5*(ru(i,k,j-1)+ru(i,k,j))
                rv_d = 0.5*(rv(i,k,j+1)+rv(i,k,j))
                ww_d = 0.5*(ww(i,k,j-1)+ww(i,k,j))
                call get_staggered_from_fluxes(vx, vy, vz,                 &
                                               fvx_adv, fvy_adv, fvz_adv,  &
                                               ru_d, rv_d, ww_d, v_save,        &
                                               cf1, cf2, cf3, cfn, cfn1, .false., &
                                               i, j, k,                           &
                                               ids, ide, jds, jde, kds, kde,      &
                                               ims, ime, jms, jme, kms, kme       )

                rho_s = 0.5*(rho8w(i,k,j-1) + rho8w(i,k,j-1))
                !for cartesian corrections
                rzxu = rho_s*0.5*(zxu(i,k,j-1)+zxu(i,k,j))
                rzyv = rho_s*0.5*(zyv(i,k,j-1)+zyv(i,k,j))
                rdzdt = rho_s*0.5*(dzdt(i,k,j-1)+dzdt(i,k,j))
                rw_diag_s = 0.5*(rw_diag(i,k,j) + rw_diag(i,k,j-1))

                !build averages
                VX_MEAN(i,k,j) = (local_count*VX_MEAN(i,k,j) + vx*rhoxy )/(local_count+1.)/rhoxy_mean
                VY_MEAN(i,k,j) = (local_count*VY_MEAN(i,k,j) + vy*rhom  )/(local_count+1.)/rho_mean
                VZ_MEAN(i,k,j) = (local_count*VZ_MEAN(i,k,j) + vz*rhoyz )/(local_count+1.)/rhoyz_mean
                FVX_ADV_MEAN(i,k,j)  = (local_count*FVX_ADV_MEAN(i,k,j) + fvx_adv(i,k,j) )/(local_count+1.)*fvx_dfac
                FVY_ADV_MEAN(i,k,j)  = (local_count*FVY_ADV_MEAN(i,k,j) + fvy_adv(i,k,j) )/(local_count+1.)*fvy_dfac
                FVZ_ADV_MEAN(i,k,j)  = (local_count*FVZ_ADV_MEAN(i,k,j) + rw_diag_s*vz )/(local_count+1.)/rhoyz_mean_flx
                FVX_CORR_MEAN(i,k,j)  = (local_count*FVX_CORR_MEAN(i,k,j) + rzxu*vz )/(local_count+1.)
                FVY_CORR_MEAN(i,k,j)  = (local_count*FVY_CORR_MEAN(i,k,j) + rzyv*vz )/(local_count+1.)
                CORR_DVDT_MEAN(i,k,j)  = (local_count*CORR_DVDT_MEAN(i,k,j) + rdzdt*vz )/(local_count+1.)

                IF (config_flags%output_v_fluxes_add == 1) THEN
                    !2nd order and prognostic w fluxes
                    vx2 = 0.5*(v_save(i-1,k,j) + v_save(i,k,j))
                    vy2 = 0.5*(v_save(i,k,j+1) + v_save(i,k,j))
                    IF (( k .gt. kts) .and. (k .lt. kte)) THEN
                      vz2 = (fnp(k)* v_save(i,k-1,j) + fnm(k)* v_save(i,k,j))
                    ELSE
                      vz2 = vz
                    ENDIF
                    w_s = 0.5*(w_save(i,k,j-1)+w_save(i,k,j)) !TODOm: boundaries available?
                    FVX_ADV_MEAN_2ND(i,k,j) = (local_count*FVX_ADV_MEAN_2ND(i,k,j) + ( ru_d*vx2 ) )/(local_count+1.)*fvx_dfac
                    FVY_ADV_MEAN_2ND(i,k,j) = (local_count*FVY_ADV_MEAN_2ND(i,k,j) + ( rv_d*vy2 ) )/(local_count+1.)*fvy_dfac
                    FVZ_ADV_MEAN_2ND_PROG(i,k,j) = (local_count*FVZ_ADV_MEAN_2ND_PROG(i,k,j) + w_s*vz2*rho_s )/(local_count+1.)/rhoyz_mean_flx
                    FVZ_ADV_MEAN_PROG(i,k,j) = (local_count*FVZ_ADV_MEAN_PROG(i,k,j) + w_s*vz*rho_s )/(local_count+1.)/rhoyz_mean_flx
                    FVZ_ADV_MEAN_2ND(i,k,j) = (local_count*FVZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*vz2 )/(local_count+1.)/rhoyz_mean_flx
                    FVX_CORR_MEAN_2ND(i,k,j) = (local_count*FVX_CORR_MEAN_2ND(i,k,j) + rzxu*vz2 )/(local_count+1.)
                    FVY_CORR_MEAN_2ND(i,k,j) = (local_count*FVY_CORR_MEAN_2ND(i,k,j) + rzyv*vz2 )/(local_count+1.)
                    CORR_DVDT_MEAN_2ND(i,k,j) = (local_count*CORR_DVDT_MEAN_2ND(i,k,j) + rdzdt*vz2 )/(local_count+1.)
                ENDIF
            ENDIF

            IF (config_flags%output_wres_fluxes) THEN
                IF (k .eq. kte) THEN
                    ru_d = (2.-fnm(k-1))*ru(i,k-1,j)-fnp(k-1)*ru(i,k-2,j)
                    rv_d = (2.-fnm(k-1))*rv(i,k-1,j)-fnp(k-1)*rv(i,k-2,j)
                    ww_d = 0
                ELSE
                    IF (k .eq. kts) THEN
                        !set to arbitrary value; not needed
                        ru_d = 1
                        rv_d = 1
                    ELSE
                        ru_d = fnm(k)*ru(i,k,j)+fnp(k)*ru(i,k-1,j)
                        rv_d = fnm(k)*rv(i,k,j)+fnp(k)*rv(i,k-1,j)
                    ENDIF
                    ww_d = 0.5*(ww(i,k,j)+ww(i,k+1,j))
                ENDIF

                call get_staggered_from_fluxes(wx, wy, wz,                 &
                                               fwx_adv, fwy_adv, fwz_adv,  &
                                               ru_d, rv_d, ww_d, w_save,        &
                                               cf1, cf2, cf3, cfn, cfn1, .true.,  &
                                               i, j, k,                           &
                                               ids, ide, jds, jde, kds, kde,      &
                                               ims, ime, jms, jme, kms, kme       )

                IF (k .gt. kts) THEN
                    WX_MEAN(i,k,j) = (local_count*WX_MEAN(i,k,j) + wx*rhoxz )/(local_count+1.)/rhoxz_mean
                    WY_MEAN(i,k,j) = (local_count*WY_MEAN(i,k,j) + wy*rhoyz )/(local_count+1.)/rhoyz_mean
                    FWX_ADV_MEAN(i,k,j)  = (local_count*FWX_ADV_MEAN(i,k,j) + fwx_adv(i,k,j) )/(local_count+1.)*fwx_dfac
                    FWY_ADV_MEAN(i,k,j)  = (local_count*FWY_ADV_MEAN(i,k,j) + fwy_adv(i,k,j) )/(local_count+1.)*fwy_dfac
                ENDIF
                IF (k .lt. kte) THEN
                    !destagger in z direction
                    rho_s = rho_d(i,k,j)
                    rzxu = rho_s*0.5*(zxu(i,k+1,j)+zxu(i,k,j))
                    rzyv = rho_s*0.5*(zyv(i,k+1,j)+zyv(i,k,j))
                    rdzdt = rho_s*0.5*(dzdt(i,k+1,j)+dzdt(i,k,j))
                    rw_diag_s = 0.5*(rw_diag(i,k+1,j) + rw_diag(i,k,j))
                    WZ_MEAN(i,k,j) = (local_count*WZ_MEAN(i,k,j) + wz*rhom  )/(local_count+1.)/rho_mean
                    FWZ_ADV_MEAN(i,k,j)  = (local_count*FWZ_ADV_MEAN(i,k,j) + rw_diag_s*wz )/(local_count+1.)/rho_mean_flx
                    FWX_CORR_MEAN(i,k,j)  = (local_count*FWX_CORR_MEAN(i,k,j) + rzxu*wz )/(local_count+1.)
                    FWY_CORR_MEAN(i,k,j)  = (local_count*FWY_CORR_MEAN(i,k,j) + rzyv*wz )/(local_count+1.)
                    CORR_DWDT_MEAN(i,k,j)  = (local_count*CORR_DWDT_MEAN(i,k,j) + rdzdt*wz )/(local_count+1.)
                ENDIF

                IF (config_flags%output_w_fluxes_add == 1) THEN
                    !2nd order and prognostic w fluxes
                    IF (k .gt. kts) THEN
                        wx2 = 0.5*(w_save(i-1,k,j) + w_save(i,k,j))
                        wy2 = 0.5*(w_save(i,k,j-1) + w_save(i,k,j))
                        FWX_ADV_MEAN_2ND(i,k,j) = (local_count*FWX_ADV_MEAN_2ND(i,k,j) + ru_d*wx2 )/(local_count+1.)*fwx_dfac
                        FWY_ADV_MEAN_2ND(i,k,j) = (local_count*FWY_ADV_MEAN_2ND(i,k,j) + rv_d*wy2 )/(local_count+1.)*fwy_dfac
                    ENDIF
                    IF (k .lt. kte) THEN
                        IF (k .lt. kte) THEN
                            wz2 = 0.5*(w_save(i,k+1,j) + w_save(i,k,j))
                        ELSE
                            wz2 = wz
                        ENDIF
                        w_s = 0.5*(w_save(i,k+1,j)+w_save(i,k,j))
                        FWZ_ADV_MEAN_2ND_PROG(i,k,j) = (local_count*FWZ_ADV_MEAN_2ND_PROG(i,k,j) + w_s*wz2*rho_s )/(local_count+1.)/rho_mean_flx
                        FWZ_ADV_MEAN_PROG(i,k,j) = (local_count*FWZ_ADV_MEAN_PROG(i,k,j) + w_s*wz*rho_s )/(local_count+1.)/rho_mean_flx
                        FWZ_ADV_MEAN_2ND(i,k,j) = (local_count*FWZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*wz2 )/(local_count+1.)/rho_mean_flx
                        FWX_CORR_MEAN_2ND(i,k,j) = (local_count*FWX_CORR_MEAN_2ND(i,k,j) + rzxu*wz2 )/(local_count+1.)
                        FWY_CORR_MEAN_2ND(i,k,j) = (local_count*FWY_CORR_MEAN_2ND(i,k,j) + rzyv*wz2 )/(local_count+1.)
                        CORR_DWDT_MEAN_2ND(i,k,j) = (local_count*CORR_DWDT_MEAN_2ND(i,k,j) + rdzdt*wz2 )/(local_count+1.)
                    ENDIF
                ENDIF
            ENDIF

            !average wind speed
            !TODOm: use mu for hesselberg averaging?
            U_MEAN(i,k,j) = (local_count*U_MEAN(i,k,j) + U(i,k,j)*rhox  )/(local_count+1.)/rhox_mean
            V_MEAN(i,k,j) = (local_count*V_MEAN(i,k,j) + V(i,k,j)*rhoy  )/(local_count+1.)/rhoy_mean
            ZWIND_MEAN(i,k,j) = (local_count*ZWIND_MEAN(i,k,j) + W(i,k,j)*rhoz  )/(local_count+1.)/rhoz_mean
            WW_MEAN(i,k,j) = (local_count*WW_MEAN(i,k,j) + ww(i,k,j))/(local_count+1.)
            WD_MEAN(i,k,j) = (local_count*WD_MEAN(i,k,j) + w_diag(i,k,j)*rhoz )/(local_count+1.)/rhoz_mean
            DPH_X_MEAN(i,k,j) = (local_count*DPH_X_MEAN(i,k,j) + dph_x(i,k,j)*rhoz )/(local_count+1.)/rhoz_mean
            DPH_Y_MEAN(i,k,j) = (local_count*DPH_Y_MEAN(i,k,j) + dph_y(i,k,j)*rhoz )/(local_count+1.)/rhoz_mean
          ENDIF

          IF (config_flags%output_t_fluxes .eq. 1) THEN
              T_TEND_RADLW_MEAN(i,k,j)  = (local_count*T_TEND_RADLW_MEAN(i,k,j) + t_tend_radlw(i,k,j) )/(local_count+1.)
              T_TEND_RADSW_MEAN(i,k,j)  = (local_count*T_TEND_RADSW_MEAN(i,k,j) + t_tend_radsw(i,k,j) )/(local_count+1.)
              T_TEND_MP_MEAN(i,k,j)  = (local_count*T_TEND_MP_MEAN(i,k,j) + t_tend_mp(i,k,j) )/(local_count+1.)
          ENDIF
          IF (config_flags%output_q_fluxes .eq. 1) THEN
              Q_TEND_MP_MEAN(i,k,j)  = (local_count*Q_TEND_MP_MEAN(i,k,j) + qv_tend_mp(i,k,j) )/(local_count+1.)
          ENDIF
          IF (config_flags%output_u_fluxes .eq. 1) THEN
              U_TEND_PG_MEAN(i,k,j)  = (local_count*U_TEND_PG_MEAN(i,k,j) + u_tend_pg(i,k,j) )/(local_count+1.)
              U_TEND_COR_CURV_MEAN(i,k,j)  = (local_count*U_TEND_COR_CURV_MEAN(i,k,j) + u_tend_cor_curv(i,k,j) )/(local_count+1.)
          ENDIF
          IF (config_flags%output_v_fluxes .eq. 1) THEN
              V_TEND_PG_MEAN(i,k,j)  = (local_count*V_TEND_PG_MEAN(i,k,j) + v_tend_pg(i,k,j) )/(local_count+1.)
              V_TEND_COR_CURV_MEAN(i,k,j)  = (local_count*V_TEND_COR_CURV_MEAN(i,k,j) + v_tend_cor_curv(i,k,j) )/(local_count+1.)
          ENDIF
          IF (config_flags%output_w_fluxes .eq. 1) THEN
              W_TEND_PG_MEAN(i,k,j)  = (local_count*W_TEND_PG_MEAN(i,k,j) + w_tend_pg(i,k,j) )/(local_count+1.)
              W_TEND_COR_CURV_MEAN(i,k,j)  = (local_count*W_TEND_COR_CURV_MEAN(i,k,j) + w_tend_cor_curv(i,k,j) )/(local_count+1.)
          ENDIF

          IF (config_flags%output_sgs_fluxes) THEN
            IF (config_flags%output_tsgs_fluxes) THEN
                FTX_SGS_MEAN(i,k,j) = (local_count*FTX_SGS_MEAN(i,k,j) + ftx_sgs(i,k,j) ) / (local_count+1.) / rhox_mean_flx
                FTY_SGS_MEAN(i,k,j) = (local_count*FTY_SGS_MEAN(i,k,j) + fty_sgs(i,k,j) ) / (local_count+1.) / rhoy_mean_flx
                FTZ_SGS_MEAN(i,k,j) = (local_count*FTZ_SGS_MEAN(i,k,j) + ftz_sgs(i,k,j) ) / (local_count+1.) / rhoz_mean_flx
            ENDIF
            IF (config_flags%output_qsgs_fluxes) THEN
                FQX_SGS_MEAN(i,k,j) = (local_count*FQX_SGS_MEAN(i,k,j) + fqx_sgs(i,k,j) ) / (local_count+1.) / rhox_mean_flx
                FQY_SGS_MEAN(i,k,j) = (local_count*FQY_SGS_MEAN(i,k,j) + fqy_sgs(i,k,j) ) / (local_count+1.) / rhoy_mean_flx
                FQZ_SGS_MEAN(i,k,j) = (local_count*FQZ_SGS_MEAN(i,k,j) + fqz_sgs(i,k,j) ) / (local_count+1.) / rhoz_mean_flx
            ENDIF
            IF (config_flags%output_usgs_fluxes) THEN
                FUX_SGS_MEAN(i,k,j)  = (local_count*FUX_SGS_MEAN(i,k,j)  + fux_sgs(i,k,j) )  / (local_count+1.) / rho_mean_flx
                FUZ_SGS_MEAN(i,k,j)  = (local_count*FUZ_SGS_MEAN(i,k,j)  + fuz_sgs(i,k,j) )  / (local_count+1.) / rhoxz_mean_flx
            ENDIF
            IF (config_flags%output_vsgs_fluxes) THEN
                FVY_SGS_MEAN(i,k,j)  = (local_count*FVY_SGS_MEAN(i,k,j)  + fvy_sgs(i,k,j) )  / (local_count+1.) / rho_mean_flx
                FVZ_SGS_MEAN(i,k,j)  = (local_count*FVZ_SGS_MEAN(i,k,j)  + fvz_sgs(i,k,j) )  / (local_count+1.) / rhoyz_mean_flx
            ENDIF
            IF (config_flags%output_usgs_fluxes .or. config_flags%output_vsgs_fluxes) THEN
                FVX_SGS_MEAN(i,k,j)  = (local_count*FVX_SGS_MEAN(i,k,j)  + fvx_sgs(i,k,j) )  / (local_count+1.) / rhoxy_mean_flx
            ENDIF
            IF (config_flags%output_wsgs_fluxes) THEN
                FWX_SGS_MEAN(i,k,j)  = (local_count*FWX_SGS_MEAN(i,k,j)  + fwx_sgs(i,k,j) )  / (local_count+1.) / rhoxz_mean_flx
                FWY_SGS_MEAN(i,k,j)  = (local_count*FWY_SGS_MEAN(i,k,j)  + fwy_sgs(i,k,j) )  / (local_count+1.) / rhoyz_mean_flx
                FWZ_SGS_MEAN(i,k,j)  = (local_count*FWZ_SGS_MEAN(i,k,j)  + fwz_sgs(i,k,j) )  / (local_count+1.) / rho_mean_flx
            ENDIF
          ENDIF

        end DO
      end DO
    end DO
    endif

    if (config_flags%do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = (local_count*avgflx_rum(i,k,j) + ru_m(i,k,j))/(local_count+1.)
             avgflx_rvm(i,k,j) = (local_count*avgflx_rvm(i,k,j) + rv_m(i,k,j))/(local_count+1.)
             avgflx_wwm(i,k,j) = (local_count*avgflx_wwm(i,k,j) + ww_m(i,k,j))/(local_count+1.)
          end DO
       end DO
    end DO
    endif

    if (do_cu .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) &
         & .and. present(cfu1) .and. present(cfd1) .and. present(dfu1) &
         & .and. present(efu1) .and. present(dfd1) .and. present(efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = (local_count*avgflx_cfu1(i,k,j) + &
                     & cfu1(i,k,j)) / (local_count+1.)
                avgflx_cfd1(i,k,j) = (local_count*avgflx_cfd1(i,k,j) + &
                     & cfd1(i,k,j)) / (local_count+1.)
                avgflx_dfu1(i,k,j) = (local_count*avgflx_dfu1(i,k,j) + &
                     & dfu1(i,k,j)) / (local_count+1.)
                avgflx_efu1(i,k,j) = (local_count*avgflx_efu1(i,k,j) + &
                     & efu1(i,k,j)) / (local_count+1.)
                avgflx_dfd1(i,k,j) = (local_count*avgflx_dfd1(i,k,j) + &
                     & dfd1(i,k,j)) / (local_count+1.)
                avgflx_efd1(i,k,j) = (local_count*avgflx_efd1(i,k,j) + &
                     & efd1(i,k,j)) / (local_count+1.)
             end DO
          end DO
       end DO
    end if

    return
  end subroutine upd_avgflx

  subroutine stagger_z(var_stag, var,                      &
                       cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                       k, kms, kme, kde                    )

    REAL, INTENT(OUT) :: var_stag
    REAL, INTENT(IN) :: cf1, cf2, cf3, cfn, cfn1
    INTEGER, INTENT(IN) ::  k, kms, kme, kde
    REAL, DIMENSION(kms:kme) , INTENT(IN) :: var, fnm, fnp

    if ( k .eq. kde) then
      var_stag = cfn*var(kde-1)+cfn1*var(kde-2)
    elseif ( k .eq. 1) then
      var_stag = cf1*var(1)+cf2*var(2)+cf3*var(3)
    else
      var_stag = fnm(k)*var(k)+ fnp(k)*var(k-1)
    endif

  end subroutine stagger_z

  subroutine get_staggered_from_fluxes(varx, vary, varz,                &
                                       ruvar_tot, rvvar_tot, wwvar_tot, &
                                       ru, rv, ww, var,                 &
                                       cf1, cf2, cf3, cfn, cfn1, do_w,  &
                                       i, j, k,                         &
                                       ids, ide, jds, jde, kds, kde,    &
                                       ims, ime, jms, jme, kms, kme     )

    IMPLICIT NONE

    REAL, INTENT(OUT) :: varx, vary, varz
    REAL, INTENT(IN) :: ru, rv, ww, cf1, cf2, cf3, cfn, cfn1
    LOGICAL, INTENT(IN) :: do_w
    INTEGER, INTENT(IN) ::  i, j, k,                       &
                            ids, ide, jds, jde, kds, kde,  &
                            ims, ime, jms, jme, kms, kme
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
                        ruvar_tot, rvvar_tot, wwvar_tot, var

    WRITE ( wrf_err_message , FMT=* ) "i,j,k:",i,",",j,",",k
    if (ru .ne. 0) then
        varx = ruvar_tot(i,k,j)/ru
    else
        varx = var(i,k,j)
        if (k .lt. kde) then
            CALL wrf_debug ( 100 , ' Found zero ru when getting staggered variable from flux! Using var(i,k,j)...' )
            CALL wrf_debug ( 100 , wrf_err_message )
        endif
    endif
    if (rv .ne. 0) then
        vary = rvvar_tot(i,k,j)/rv
    else
        vary = var(i,k,j)
        if (k .lt. kde) then
            CALL wrf_debug ( 100 , ' Found zero rv when getting staggered variable from flux! Using var(i,k,j)...' )
            CALL wrf_debug ( 100 , wrf_err_message )
        endif
    endif

    if ((do_w) .and. (k .eq. kde)) THEN
      varz = 0
    elseif (k .eq. kde) then
      varz = cfn*var(i,kde-1,j)+cfn1*var(i,kde-2,j)
    elseif ((k .eq. 1) .and. (.not. do_w)) then
      varz = cf1*var(i,1,j)+cf2*var(i,2,j)+cf3*var(i,3,j)
    elseif (ww .ne. 0) then
      varz = wwvar_tot(i,k,j)/ww
    else
      varz = var(i,k,j)
      IF ((i .lt. ide) .and. (j .lt. jde)) THEN
        CALL wrf_debug ( 100 , ' Found zero ww when getting staggered variable from flux! Using var(i,k,j)...' )
        CALL wrf_debug ( 100 , wrf_err_message )
      ENDIF
    endif
  end subroutine get_staggered_from_fluxes

end MODULE module_avgflx_em
