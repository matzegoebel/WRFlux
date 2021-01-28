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
       & ftx_adv_mean_2nd,fty_adv_mean_2nd,ftz_adv_mean_2nd, ftx_corr_2nd,fty_corr_2nd,corr_dtdt_2nd, &
       & ftx_corr_dzout,fty_corr_dzout,                                                               &
       & fqx_adv_mean_2nd,fqy_adv_mean_2nd,fqz_adv_mean_2nd, fqx_corr_2nd,fqy_corr_2nd,corr_dqdt_2nd, &
       & fqx_corr_dzout,fqy_corr_dzout,                                                               &
       & fux_adv_mean_2nd,fuy_adv_mean_2nd,fuz_adv_mean_2nd, fux_corr_2nd,fuy_corr_2nd,corr_dudt_2nd, &
       & fux_corr_dzout,fuy_corr_dzout,                                                               &
       & fvx_adv_mean_2nd,fvy_adv_mean_2nd,fvz_adv_mean_2nd, fvx_corr_2nd,fvy_corr_2nd,corr_dvdt_2nd, &
       & fvx_corr_dzout,fvy_corr_dzout,                                                               &
       & fwx_adv_mean_2nd,fwy_adv_mean_2nd,fwz_adv_mean_2nd, fwx_corr_2nd,fwy_corr_2nd,corr_dwdt_2nd, &
       & fwx_corr_dzout,fwy_corr_dzout,                                                               &
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
       ftx_adv_mean_2nd,fty_adv_mean_2nd,ftz_adv_mean_2nd, ftx_corr_2nd,fty_corr_2nd,corr_dtdt_2nd, &
       ftx_corr_dzout,fty_corr_dzout,                                                               &
       fqx_adv_mean_2nd,fqy_adv_mean_2nd,fqz_adv_mean_2nd, fqx_corr_2nd,fqy_corr_2nd,corr_dqdt_2nd, &
       fqx_corr_dzout,fqy_corr_dzout,                                                               &
       fux_adv_mean_2nd,fuy_adv_mean_2nd,fuz_adv_mean_2nd, fux_corr_2nd,fuy_corr_2nd,corr_dudt_2nd, &
       fux_corr_dzout,fuy_corr_dzout,                                                               &
       fvx_adv_mean_2nd,fvy_adv_mean_2nd,fvz_adv_mean_2nd, fvx_corr_2nd,fvy_corr_2nd,corr_dvdt_2nd, &
       fvx_corr_dzout,fvy_corr_dzout,                                                               &
       fwx_adv_mean_2nd,fwy_adv_mean_2nd,fwz_adv_mean_2nd, fwx_corr_2nd,fwy_corr_2nd,corr_dwdt_2nd, &
       fwx_corr_dzout,fwy_corr_dzout,                                                                    &
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
              ftx_corr_dzout(i,k,j) = 0.
              fty_corr_dzout(i,k,j) = 0.
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
              fqx_corr_dzout(i,k,j) = 0.
              fqy_corr_dzout(i,k,j) = 0.
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
              fux_corr_dzout(i,k,j) = 0.
              fuy_corr_dzout(i,k,j) = 0.
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
              fvx_corr_dzout(i,k,j) = 0.
              fvy_corr_dzout(i,k,j) = 0.
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
              fwx_corr_dzout(i,k,j) = 0.
              fwy_corr_dzout(i,k,j) = 0.
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
        IF (config_flags%output_res_fluxes) THEN
          dph_x_mean(i,k,j) = 0.
          dph_y_mean(i,k,j) = 0.
        ENDIF
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
       & FTX_ADV_MEAN_2ND,FTY_ADV_MEAN_2ND,FTZ_ADV_MEAN_2ND,FTX_CORR_MEAN_2ND,FTY_CORR_MEAN_2ND,CORR_DTDT_MEAN_2ND,  &
       & FTX_CORR_MEAN_DZOUT,FTY_CORR_MEAN_DZOUT,                                                                    &
       & FQX_ADV_MEAN_2ND,FQY_ADV_MEAN_2ND,FQZ_ADV_MEAN_2ND,FQX_CORR_MEAN_2ND,FQY_CORR_MEAN_2ND,CORR_DQDT_MEAN_2ND,  &
       & FQX_CORR_MEAN_DZOUT,FQY_CORR_MEAN_DZOUT,                                                                    &
       & FUX_ADV_MEAN_2ND,FUY_ADV_MEAN_2ND,FUZ_ADV_MEAN_2ND,FUX_CORR_MEAN_2ND,FUY_CORR_MEAN_2ND,CORR_DUDT_MEAN_2ND,  &
       & FUX_CORR_MEAN_DZOUT,FUY_CORR_MEAN_DZOUT,                                                                    &
       & FVX_ADV_MEAN_2ND,FVY_ADV_MEAN_2ND,FVZ_ADV_MEAN_2ND,FVX_CORR_MEAN_2ND,FVY_CORR_MEAN_2ND,CORR_DVDT_MEAN_2ND,  &
       & FVX_CORR_MEAN_DZOUT,FVY_CORR_MEAN_DZOUT,                                                                    &
       & FWX_ADV_MEAN_2ND,FWY_ADV_MEAN_2ND,FWZ_ADV_MEAN_2ND,FWX_CORR_MEAN_2ND,FWY_CORR_MEAN_2ND,CORR_DWDT_MEAN_2ND,  &
       & FWX_CORR_MEAN_DZOUT,FWY_CORR_MEAN_DZOUT,                                                                    &
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
       & u_2, u_save, v_2, v_save, w_2, w_save, ru, rv, ww, t_save, thd_save, th_phy, Q,         &
       & ph_1, ph_2, phb, zx, zy, alt, dph_x, dph_y,                                             &
       & fnm, fnp, mut, c1h, c2h, c1f, c2f, cf1, cf2, cf3, cfn, cfn1,                       &
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
       FTX_ADV_MEAN_2ND,FTY_ADV_MEAN_2ND,FTZ_ADV_MEAN_2ND,FTX_CORR_MEAN_2ND,FTY_CORR_MEAN_2ND,CORR_DTDT_MEAN_2ND,  &
       FTX_CORR_MEAN_DZOUT,FTY_CORR_MEAN_DZOUT,                                                                    &
       FQX_ADV_MEAN_2ND,FQY_ADV_MEAN_2ND,FQZ_ADV_MEAN_2ND,FQX_CORR_MEAN_2ND,FQY_CORR_MEAN_2ND,CORR_DQDT_MEAN_2ND,  &
       FQX_CORR_MEAN_DZOUT,FQY_CORR_MEAN_DZOUT,                                                                    &
       FUX_ADV_MEAN_2ND,FUY_ADV_MEAN_2ND,FUZ_ADV_MEAN_2ND,FUX_CORR_MEAN_2ND,FUY_CORR_MEAN_2ND,CORR_DUDT_MEAN_2ND,  &
       FUX_CORR_MEAN_DZOUT,FUY_CORR_MEAN_DZOUT,                                                                    &
       FVX_ADV_MEAN_2ND,FVY_ADV_MEAN_2ND,FVZ_ADV_MEAN_2ND,FVX_CORR_MEAN_2ND,FVY_CORR_MEAN_2ND,CORR_DVDT_MEAN_2ND,  &
       FVX_CORR_MEAN_DZOUT,FVY_CORR_MEAN_DZOUT,                                                                    &
       FWX_ADV_MEAN_2ND,FWY_ADV_MEAN_2ND,FWZ_ADV_MEAN_2ND,FWX_CORR_MEAN_2ND,FWY_CORR_MEAN_2ND,CORR_DWDT_MEAN_2ND,  &
       FWX_CORR_MEAN_DZOUT,FWY_CORR_MEAN_DZOUT,                                                                    &
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
         u_2, u_save, v_2, v_save, w_2, w_save, ru, rv, ww,  &
         t_save, thd_save, th_phy, Q, ph_1, ph_2, phb, zx, zy, alt,  &
         ftx_small, fty_small, ftz_small,  &
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

    REAL, OPTIONAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(INOUT) :: ftx_adv, fty_adv, ftz_adv,  &
                                                                            dph_x, dph_y

    REAL , DIMENSION( kms:kme ) , OPTIONAL, INTENT(IN   ) :: fnm, fnp, c1h, c2h, c1f, c2f
    REAL , DIMENSION( ims:ime, jms:jme ), OPTIONAL, INTENT(IN   ) :: mut
    REAL, OPTIONAL, INTENT(IN) :: dt, cf1, cf2, cf3, cfn, cfn1
    REAL, OPTIONAL, DIMENSION( ims:ime , jms:jme ) ,  INTENT(IN   ) :: msfuy,  &
                                                                       msfvx,  &
                                                                       msfvy,  &
                                                                       msftx,  &
                                                                       msfty

    LOGICAL, OPTIONAL, INTENT(IN) :: last_timestep


    REAL,    DIMENSION(ims:ime, kms:kme, jms:jme) ::    rho_d
    REAL,    DIMENSION( its-1:ite+1 , kts:kte , jts-1:jte+1 ) ::    rdzdt, u8w, v8w, rho8w, rw_diag, rzxu, rzyv

    INTEGER :: i,j,k
    REAL :: local_count, count_p1, rhom, rhox, rhoy, rhoz, rhoxy, rhoxz, rhoyz,            &
            rho_mean, rhox_mean, rhoy_mean, rhoz_mean, rhoxy_mean, rhoxz_mean, rhoyz_mean, &
            rho_mean_flx, rhox_mean_flx, rhoy_mean_flx, rhoz_mean_flx, rhoxy_mean_flx,     &
            rhoxz_mean_flx, rhoyz_mean_flx, &
            mu_mean, mux_mean, muy_mean, muxy_mean, muxz_mean, muyz_mean,        &
            fqx_dfac, fqy_dfac, &
            fux_dfac, fuy_dfac, &
            fvx_dfac, fvy_dfac, &
            fwx_dfac, fwy_dfac, &
            thetax, thetay, thetaz, qx, qy, qz, ux, uy, uz, vx, vy, vz, wx, wy, wz,     &
            ru_d, rv_d, ww_d,               &
            rzxu_s, rzyv_s, ru8z, rv8z, rdzdt_s

    REAL :: thetax2, thetay2, thetaz2,                 &
            qx2, qy2, qz2,                             &
            ux2, uy2, uz2,                             &
            vx2, vy2, vz2,                             &
            wx2, wy2, wz2,                             &
            rw_diag_s


    local_count = real(avgflx_count)
    count_p1 = local_count + 1.

    if (present(dt) .and. config_flags%output_avgfluxes) then

      DO j=jts-1,jte
      DO k=kts,min(kte,kde-1)
      DO i=its-1,ite
        !calculate dry density
        rho_d(i,k,j) = 1./alt(i,k,j)
        RHOD_MEAN(i,k,j) = (local_count*RHOD_MEAN(i,k,j)  + rho_d(i,k,j)  )/count_p1
      ENDDO
      ENDDO
      ENDDO

      k = kde
      DO j=jts-1,jte
      DO i=its-1,ite
        !set upper boundary to prevent errors
        rho_d(i,k,j) = rho_d(i,k-1,j)
        RHOD_MEAN(i,k,j) = RHOD_MEAN(i,k-1,j)
      ENDDO
      ENDDO

      DO j=jts-1,jte
      DO i=its-1  ,ite
        MUT_MEAN(i,j) = (local_count*MUT_MEAN(i,j) + mut(i,j))/count_p1
      ENDDO
      ENDDO

      IF (config_flags%output_res_fluxes) THEN
        DO j=jts-1,jte+1
        DO k=kts,kte
        DO i=its-1,ite+1
          call stagger_z(u8w(i,k,j), u_save(i,:,j),  &
                         cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                         k, kms, kme, kde              )
          call stagger_z(v8w(i,k,j), v_save(i,:,j),  &
                         cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                         k, kms, kme, kde              )
          call stagger_z(rho8w(i,k,j), rho_d(i,:,j),  &
                         cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                         k, kms, kme, kde              )
          rdzdt(i,k,j) = rho8w(i,k,j)*(ph_2(i,k,j) - ph_1(i,k,j))/(g*dt)

        ENDDO
        ENDDO
        ENDDO
        DO j=jts-1,jte
        DO k=kts,kte
        DO i=its-1,ite
          !diagnostic velocity
          IF (k .eq. kts) THEN
            dph_x(i,k,j) = -g*0.5*msftx(i,j)*(zx(i,k,j)*u8w(i,k,j) + zx(i+1,k,j)*u8w(i+1,k,j))
            dph_y(i,k,j) = -g*0.5*msfty(i,j)*(zy(i,k,j)*v8w(i,k,j) + zy(i,k,j+1)*v8w(i,k,j+1))
          ENDIF
          rzxu(i,k,j) = -rho8w(i,k,j)*dph_x(i,k,j)/g
          rzyv(i,k,j) = -rho8w(i,k,j)*dph_y(i,k,j)/g
          rw_diag(i,k,j) = rdzdt(i,k,j) + rzxu(i,k,j) + rzyv(i,k,j) + ww(i,k,j)*msfty(i,j)/(-g)
          w_diag(i,k,j) = rw_diag(i,k,j)/rho8w(i,k,j)
        ENDDO
        ENDDO
        ENDDO
      ENDIF

      rhom = 1.
      rhox = 1.
      rhoy = 1.
      rhoxy = 1.
      rhoz = 1.
      rhoxz = 1.
      rhoyz = 1.

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
      fux_dfac = 1.
      fuy_dfac = 1.
      fvx_dfac = 1.
      fvy_dfac = 1.
      fwx_dfac = 1.
      fwy_dfac = 1.

      DO j=jts,jte
      DO k=kts,kte
      DO i=its,ite
        Z_MEAN(i,k,j) = (local_count*Z_MEAN(i,k,j)  + (phb(i,k,j)+ph_2(i,k,j))/g  )/count_p1
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

          IF (config_flags%output_res_fluxes) THEN
            !factors to decouple horizontal fluxes from mu and map-scale factors
            muxz_mean = c1f(k)*0.5*(MUT_MEAN(i,j) + MUT_MEAN(i-1,j)) + c2f(k)
            muyz_mean = c1f(k)*0.5*(MUT_MEAN(i,j) + MUT_MEAN(i,j-1)) + c2f(k)
            fwx_dfac  = msfuy(i,j)/muxz_mean
            fwy_dfac  = msfvx(i,j)/muyz_mean
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
          endif

          !determine staggered variables from total fluxes
          IF (config_flags%output_tres_fluxes) THEN
            IF (config_flags%output_dry_theta_fluxes .and. (config_flags%use_theta_m .eq. 1)) THEN
              call get_staggered_from_fluxes(thetax, thetay, thetaz,      &
                                             ftx_adv, fty_adv, ftz_adv,&
                                             ru(i,k,j), rv(i,k,j), ww(i,k,j), thd_save, &
                                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp,       &
                                             .false., .false., .false.,                &
                                             i, j, k,                             &
                                             ids, ide, jds, jde, kds, kde,        &
                                             ims, ime, jms, jme, kms, kme         )
            ELSE
              call get_staggered_from_fluxes(thetax, thetay, thetaz,      &
                                             ftx_adv, fty_adv, ftz_adv,&
                                             ru(i,k,j), rv(i,k,j), ww(i,k,j), t_save, &
                                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                                            .false., .false., .false.,                &
                                             i, j, k,                             &
                                             ids, ide, jds, jde, kds, kde,        &
                                             ims, ime, jms, jme, kms, kme         )
            ENDIF

            ftz_adv(i,k,j) = rw_diag(i,k,j)*thetaz
            IF (config_flags%output_t_fluxes_small == 1) THEN
              ftx_adv(i,k,j) = ftx_adv(i,k,j) + ftx_small(i,k,j)
              fty_adv(i,k,j) = fty_adv(i,k,j) + fty_small(i,k,j)
              ftz_adv(i,k,j) = ftz_adv(i,k,j) + ftz_small(i,k,j)*msfty(i,j)/(-g)
            ENDIF

            !build averages
            IF (k .lt. kde) THEN
              T_MEAN(i,k,j) = (local_count*T_MEAN(i,k,j) + th_phy(i,k,j)*rhom )/count_p1/rho_mean
              TX_MEAN(i,k,j) = (local_count*TX_MEAN(i,k,j) + thetax*rhox )/count_p1/rhox_mean
              TY_MEAN(i,k,j) = (local_count*TY_MEAN(i,k,j) + thetay*rhoy )/count_p1/rhoy_mean
              TT_MEAN(i,k,j) = (local_count*TT_MEAN(i,k,j) + (th_phy(i,k,j) - t0)**2 )/count_p1
              FTX_ADV_MEAN(i,k,j)  = (local_count*FTX_ADV_MEAN(i,k,j) + ftx_adv(i,k,j) )/count_p1*fqx_dfac
              FTY_ADV_MEAN(i,k,j)  = (local_count*FTY_ADV_MEAN(i,k,j) + fty_adv(i,k,j) )/count_p1*fqy_dfac
            ENDIF

            TZ_MEAN(i,k,j) = (local_count*TZ_MEAN(i,k,j) + thetaz*rhoz )/count_p1/rhoz_mean
            FTZ_ADV_MEAN(i,k,j)  = (local_count*FTZ_ADV_MEAN(i,k,j) + ftz_adv(i,k,j) )/count_p1/rhoz_mean_flx
            FTX_CORR_MEAN(i,k,j)  = (local_count*FTX_CORR_MEAN(i,k,j) + rzxu(i,k,j)*thetaz )/count_p1
            FTY_CORR_MEAN(i,k,j)  = (local_count*FTY_CORR_MEAN(i,k,j) + rzyv(i,k,j)*thetaz )/count_p1
            CORR_DTDT_MEAN(i,k,j)  = (local_count*CORR_DTDT_MEAN(i,k,j) + rdzdt(i,k,j)*thetaz )/count_p1

            IF (config_flags%output_t_fluxes_add == 1) THEN
              !2nd order fluxes
              IF (config_flags%output_dry_theta_fluxes .and. (config_flags%use_theta_m .eq. 1)) THEN
                thetax2 = 0.5*(thd_save(i-1,k,j) + thd_save(i,k,j))
                thetay2 = 0.5*(thd_save(i,k,j-1) + thd_save(i,k,j))
                call stagger_z(thetaz2, thd_save(i,:,j),           &
                               cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                               k, kms, kme, kde                   )
              ELSE
                thetax2 = 0.5*(t_save(i-1,k,j) + t_save(i,k,j))
                thetay2 = 0.5*(t_save(i,k,j-1) + t_save(i,k,j))
                call stagger_z(thetaz2, t_save(i,:,j),             &
                               cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                               k, kms, kme, kde                   )
              ENDIF
              ru8z = rhoz*0.5*(u8w(i,k,j) + u8w(i+1,k,j))
              rv8z = rhoz*0.5*(v8w(i,k,j) + v8w(i,k,j+1))
              FTX_ADV_MEAN_2ND(i,k,j) = (local_count*FTX_ADV_MEAN_2ND(i,k,j)+ (ru(i,k,j)*thetax2 ) )/count_p1*fqx_dfac
              FTY_ADV_MEAN_2ND(i,k,j) = (local_count*FTY_ADV_MEAN_2ND(i,k,j)+ (rv(i,k,j)*thetay2 ) )/count_p1*fqy_dfac
              FTZ_ADV_MEAN_2ND(i,k,j) = (local_count*FTZ_ADV_MEAN_2ND(i,k,j)+ (rw_diag(i,k,j)*thetaz2))/count_p1/rhoz_mean_flx
              FTX_CORR_MEAN_DZOUT(i,k,j) = (local_count*FTX_CORR_MEAN_DZOUT(i,k,j) + ru8z*thetaz )/count_p1
              FTY_CORR_MEAN_DZOUT(i,k,j) = (local_count*FTY_CORR_MEAN_DZOUT(i,k,j) + rv8z*thetaz )/count_p1
              FTX_CORR_MEAN_2ND(i,k,j) = (local_count*FTX_CORR_MEAN_2ND(i,k,j) + rzxu(i,k,j)*thetaz2 )/count_p1
              FTY_CORR_MEAN_2ND(i,k,j) = (local_count*FTY_CORR_MEAN_2ND(i,k,j) + rzyv(i,k,j)*thetaz2 )/count_p1
              CORR_DTDT_MEAN_2ND(i,k,j) = (local_count*CORR_DTDT_MEAN_2ND(i,k,j) + rdzdt(i,k,j)*thetaz2 )/count_p1
            ENDIF
          ENDIF

          IF (config_flags%output_qres_fluxes) THEN
            !staggered q, different ru,rv,ww needed
            call get_staggered_from_fluxes(qx, qy, qz,                 &
                                           fqx_adv, fqy_adv, fqz_adv,  &
                                           ru_m(i,k,j), rv_m(i,k,j), ww_m(i,k,j), Q, &
                                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp,       &
                                           .false., .false., .false.,                &
                                           i, j, k,                                  &
                                           ids, ide, jds, jde, kds, kde,             &
                                           ims, ime, jms, jme, kms, kme              )

            !use ww_m instead of ww
            rw_diag_s = rdzdt(i,k,j) + rzxu(i,k,j) + rzyv(i,k,j) + ww_m(i,k,j)*msfty(i,j)/(-g)

            !build averages
            IF (k .lt. kde) THEN
              Q_MEAN(i,k,j) = (local_count*Q_MEAN(i,k,j) + Q(i,k,j)*rhom )/count_p1/rho_mean
              QX_MEAN(i,k,j) = (local_count*QX_MEAN(i,k,j) + qx*rhox )/count_p1/rhox_mean
              QY_MEAN(i,k,j) = (local_count*QY_MEAN(i,k,j) + qy*rhoy )/count_p1/rhoy_mean
              QQ_MEAN(i,k,j) = (local_count*Q_MEAN(i,k,j) + Q(i,k,j)**2 )/count_p1
              FQX_ADV_MEAN(i,k,j)  = (local_count*FQX_ADV_MEAN(i,k,j) + fqx_adv(i,k,j) )/count_p1*fqx_dfac
              FQY_ADV_MEAN(i,k,j)  = (local_count*FQY_ADV_MEAN(i,k,j) + fqy_adv(i,k,j) )/count_p1*fqy_dfac
            ENDIF
            QZ_MEAN(i,k,j) = (local_count*QZ_MEAN(i,k,j) + qz*rhoz )/count_p1/rhoz_mean
            FQZ_ADV_MEAN(i,k,j)  = (local_count*FQZ_ADV_MEAN(i,k,j) + rw_diag_s*qz )/count_p1/rhoz_mean_flx
            FQX_CORR_MEAN(i,k,j)  = (local_count*FQX_CORR_MEAN(i,k,j) + rzxu(i,k,j)*qz )/count_p1
            FQY_CORR_MEAN(i,k,j)  = (local_count*FQY_CORR_MEAN(i,k,j) + rzyv(i,k,j)*qz )/count_p1
            CORR_DQDT_MEAN(i,k,j)  = (local_count*CORR_DQDT_MEAN(i,k,j) + rdzdt(i,k,j)*qz )/count_p1

            IF (config_flags%output_q_fluxes_add == 1) THEN
              !2nd order fluxes
              qx2 = 0.5*(Q(i-1,k,j) + Q(i,k,j))
              qy2 = 0.5*(Q(i,k,j-1) + Q(i,k,j))
              IF (( k .gt. kts) .and. (k .lt. kte)) THEN
                qz2 = (fnp(k)* Q(i,k-1,j) + fnm(k)* Q(i,k,j))
              ELSE
                qz2 = qz
              ENDIF
              ru8z = rhoz*0.5*(u8w(i,k,j) + u8w(i+1,k,j))
              rv8z = rhoz*0.5*(v8w(i,k,j) + v8w(i,k,j+1))
              FQX_ADV_MEAN_2ND(i,k,j) = (local_count*FQX_ADV_MEAN_2ND(i,k,j) + ( ru_m(i,k,j)*qx2 ) )/count_p1*fqx_dfac
              FQY_ADV_MEAN_2ND(i,k,j) = (local_count*FQY_ADV_MEAN_2ND(i,k,j) + ( rv_m(i,k,j)*qy2 ) )/count_p1*fqy_dfac
              FQZ_ADV_MEAN_2ND(i,k,j) = (local_count*FQZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*qz2 )/count_p1/rhoz_mean_flx
              FQX_CORR_MEAN_DZOUT(i,k,j) = (local_count*FQX_CORR_MEAN_DZOUT(i,k,j) + ru8z*qz )/count_p1
              FQY_CORR_MEAN_DZOUT(i,k,j) = (local_count*FQY_CORR_MEAN_DZOUT(i,k,j) + rv8z*qz )/count_p1
              FQX_CORR_MEAN_2ND(i,k,j) = (local_count*FQX_CORR_MEAN_2ND(i,k,j) + rzxu(i,k,j)*qz2 )/count_p1
              FQY_CORR_MEAN_2ND(i,k,j) = (local_count*FQY_CORR_MEAN_2ND(i,k,j) + rzyv(i,k,j)*qz2 )/count_p1
              CORR_DQDT_MEAN_2ND(i,k,j) = (local_count*CORR_DQDT_MEAN_2ND(i,k,j) + rdzdt(i,k,j)*qz2 )/count_p1
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
                                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp,&
                                           .true., .false., .false.,          &
                                           i, j, k,                           &
                                           ids, ide, jds, jde, kds, kde,      &
                                           ims, ime, jms, jme, kms, kme       )

            !stagger in x direction for cartesian corrections
            rzxu_s = 0.5*(rzxu(i,k,j)/msfty(i,j) + rzxu(i-1,k,j)/msfty(i-1,j))*msfuy(i,j)
            rzyv_s = 0.5*(rzyv(i,k,j)/msfty(i,j) + rzyv(i-1,k,j)/msfty(i-1,j))*msfuy(i,j)
            rdzdt_s = 0.5*(rdzdt(i,k,j)/msfty(i,j) + rdzdt(i-1,k,j)/msfty(i-1,j))*msfuy(i,j)
            rw_diag_s = 0.5*(rw_diag(i,k,j)/msfty(i,j) + rw_diag(i-1,k,j)/msfty(i-1,j))*msfuy(i,j)

            !build averages
            UX_MEAN(i,k,j) = (local_count*UX_MEAN(i,k,j) + ux*rhom  )/count_p1/rho_mean
            UY_MEAN(i,k,j) = (local_count*UY_MEAN(i,k,j) + uy*rhoxy )/count_p1/rhoxy_mean
            UZ_MEAN(i,k,j) = (local_count*UZ_MEAN(i,k,j) + uz*rhoxz )/count_p1/rhoxz_mean
            FUX_ADV_MEAN(i,k,j)  = (local_count*FUX_ADV_MEAN(i,k,j) + fux_adv(i,k,j) )/count_p1*fux_dfac
            FUY_ADV_MEAN(i,k,j)  = (local_count*FUY_ADV_MEAN(i,k,j) + fuy_adv(i,k,j) )/count_p1*fuy_dfac
            FUZ_ADV_MEAN(i,k,j)  = (local_count*FUZ_ADV_MEAN(i,k,j) + rw_diag_s*uz )/count_p1/rhoxz_mean_flx
            FUX_CORR_MEAN(i,k,j)  = (local_count*FUX_CORR_MEAN(i,k,j) + rzxu_s*uz )/count_p1
            FUY_CORR_MEAN(i,k,j)  = (local_count*FUY_CORR_MEAN(i,k,j) + rzyv_s*uz )/count_p1
            CORR_DUDT_MEAN(i,k,j)  = (local_count*CORR_DUDT_MEAN(i,k,j) + rdzdt_s*uz )/count_p1

            IF (config_flags%output_u_fluxes_add == 1) THEN
              !2nd order fluxes
              ux2 = 0.5*(u_save(i+1,k,j) + u_save(i,k,j))
              uy2 = 0.5*(u_save(i,k,j-1) + u_save(i,k,j))
              IF (( k .gt. kts) .and. (k .lt. kte)) THEN
                uz2 = (fnp(k)* u_save(i,k-1,j) + fnm(k)* u_save(i,k,j))
              ELSE
                uz2 = uz
              ENDIF
              ru8z = rhoxz*u8w(i,k,j)
              rv8z = rhoxz*0.25*(v8w(i,k,j) + v8w(i,k,j+1) + v8w(i-1,k,j) + v8w(i-1,k,j+1))
              FUX_ADV_MEAN_2ND(i,k,j) = (local_count*FUX_ADV_MEAN_2ND(i,k,j) + ru_d*ux2 )/count_p1*fux_dfac
              FUY_ADV_MEAN_2ND(i,k,j) = (local_count*FUY_ADV_MEAN_2ND(i,k,j) + rv_d*uy2 )/count_p1*fuy_dfac
              FUZ_ADV_MEAN_2ND(i,k,j) = (local_count*FUZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*uz2 )/count_p1/rhoxz_mean_flx
              FUX_CORR_MEAN_DZOUT(i,k,j) = (local_count*FUX_CORR_MEAN_DZOUT(i,k,j) + ru8z*uz )/count_p1
              FUY_CORR_MEAN_DZOUT(i,k,j) = (local_count*FUY_CORR_MEAN_DZOUT(i,k,j) + rv8z*uz )/count_p1
              FUX_CORR_MEAN_2ND(i,k,j) = (local_count*FUX_CORR_MEAN_2ND(i,k,j) + rzxu_s*uz2 )/count_p1
              FUY_CORR_MEAN_2ND(i,k,j) = (local_count*FUY_CORR_MEAN_2ND(i,k,j) + rzyv_s*uz2 )/count_p1
              CORR_DUDT_MEAN_2ND(i,k,j) = (local_count*CORR_DUDT_MEAN_2ND(i,k,j) + rdzdt_s*uz2 )/count_p1
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
                                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp,&
                                           .false., .true., .false.,          &
                                           i, j, k,                           &
                                           ids, ide, jds, jde, kds, kde,      &
                                           ims, ime, jms, jme, kms, kme       )

            !stagger in y direction for cartesian corrections
            rzxu_s = 0.5*(rzxu(i,k,j)/msfty(i,j) + rzxu(i,k,j-1)/msfty(i,j-1))*msfvy(i,j)
            rzyv_s = 0.5*(rzyv(i,k,j)/msfty(i,j) + rzyv(i,k,j-1)/msfty(i,j-1))*msfvy(i,j)
            rdzdt_s = 0.5*(rdzdt(i,k,j)/msfty(i,j) + rdzdt(i,k,j-1)/msfty(i,j-1))*msfvy(i,j)
            rw_diag_s = 0.5*(rw_diag(i,k,j)/msfty(i,j) + rw_diag(i,k,j-1)/msfty(i,j-1))*msfvy(i,j)

            !build averages
            VX_MEAN(i,k,j) = (local_count*VX_MEAN(i,k,j) + vx*rhoxy )/count_p1/rhoxy_mean
            VY_MEAN(i,k,j) = (local_count*VY_MEAN(i,k,j) + vy*rhom  )/count_p1/rho_mean
            VZ_MEAN(i,k,j) = (local_count*VZ_MEAN(i,k,j) + vz*rhoyz )/count_p1/rhoyz_mean
            FVX_ADV_MEAN(i,k,j)  = (local_count*FVX_ADV_MEAN(i,k,j) + fvx_adv(i,k,j) )/count_p1*fvx_dfac
            FVY_ADV_MEAN(i,k,j)  = (local_count*FVY_ADV_MEAN(i,k,j) + fvy_adv(i,k,j) )/count_p1*fvy_dfac
            FVZ_ADV_MEAN(i,k,j)  = (local_count*FVZ_ADV_MEAN(i,k,j) + rw_diag_s*vz )/count_p1/rhoyz_mean_flx
            FVX_CORR_MEAN(i,k,j)  = (local_count*FVX_CORR_MEAN(i,k,j) + rzxu_s*vz )/count_p1
            FVY_CORR_MEAN(i,k,j)  = (local_count*FVY_CORR_MEAN(i,k,j) + rzyv_s*vz )/count_p1
            CORR_DVDT_MEAN(i,k,j)  = (local_count*CORR_DVDT_MEAN(i,k,j) + rdzdt_s*vz )/count_p1

            IF (config_flags%output_v_fluxes_add == 1) THEN
              !2nd order fluxes
              vx2 = 0.5*(v_save(i-1,k,j) + v_save(i,k,j))
              vy2 = 0.5*(v_save(i,k,j+1) + v_save(i,k,j))
              IF (( k .gt. kts) .and. (k .lt. kte)) THEN
                vz2 = (fnp(k)* v_save(i,k-1,j) + fnm(k)* v_save(i,k,j))
              ELSE
                vz2 = vz
              ENDIF
              ru8z = rhoyz*0.25*(u8w(i,k,j) + u8w(i+1,k,j) + u8w(i,k,j-1) + u8w(i+1,k,j-1))
              rv8z = rhoyz*v8w(i,k,j)
              FVX_ADV_MEAN_2ND(i,k,j) = (local_count*FVX_ADV_MEAN_2ND(i,k,j) + ( ru_d*vx2 ) )/count_p1*fvx_dfac
              FVY_ADV_MEAN_2ND(i,k,j) = (local_count*FVY_ADV_MEAN_2ND(i,k,j) + ( rv_d*vy2 ) )/count_p1*fvy_dfac
              FVZ_ADV_MEAN_2ND(i,k,j) = (local_count*FVZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*vz2 )/count_p1/rhoyz_mean_flx
              FVX_CORR_MEAN_DZOUT(i,k,j) = (local_count*FVX_CORR_MEAN_DZOUT(i,k,j) + ru8z*vz )/count_p1
              FVY_CORR_MEAN_DZOUT(i,k,j) = (local_count*FVY_CORR_MEAN_DZOUT(i,k,j) + rv8z*vz )/count_p1
              FVX_CORR_MEAN_2ND(i,k,j) = (local_count*FVX_CORR_MEAN_2ND(i,k,j) + rzxu_s*vz2 )/count_p1
              FVY_CORR_MEAN_2ND(i,k,j) = (local_count*FVY_CORR_MEAN_2ND(i,k,j) + rzyv_s*vz2 )/count_p1
              CORR_DVDT_MEAN_2ND(i,k,j) = (local_count*CORR_DVDT_MEAN_2ND(i,k,j) + rdzdt_s*vz2 )/count_p1
            ENDIF
          ENDIF

          IF (config_flags%output_wres_fluxes) THEN
            IF (k .lt. kte) THEN
              IF (k .gt. kts) THEN
                ru_d = fnm(k)*ru(i,k,j)+fnp(k)*ru(i,k-1,j)
                rv_d = fnm(k)*rv(i,k,j)+fnp(k)*rv(i,k-1,j)
              ELSE
                !set to arbitrary value; not needed
                ru_d = 1
                rv_d = 1
              ENDIF
              ww_d = 0.5*(ww(i,k,j)+ww(i,k+1,j))
            ELSE
              ru_d = (2.-fnm(k-1))*ru(i,k-1,j)-fnp(k-1)*ru(i,k-2,j)
              rv_d = (2.-fnm(k-1))*rv(i,k-1,j)-fnp(k-1)*rv(i,k-2,j)
              ww_d = 1 !not needed
            ENDIF

            call get_staggered_from_fluxes(wx, wy, wz,                 &
                                           fwx_adv, fwy_adv, fwz_adv,  &
                                           ru_d, rv_d, ww_d, w_save,        &
                                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp,&
                                           .false., .false., .true.,          &
                                           i, j, k,                           &
                                           ids, ide, jds, jde, kds, kde,      &
                                           ims, ime, jms, jme, kms, kme       )

            IF (k .gt. kts) THEN
              WX_MEAN(i,k,j) = (local_count*WX_MEAN(i,k,j) + wx*rhoxz )/count_p1/rhoxz_mean
              WY_MEAN(i,k,j) = (local_count*WY_MEAN(i,k,j) + wy*rhoyz )/count_p1/rhoyz_mean
              FWX_ADV_MEAN(i,k,j)  = (local_count*FWX_ADV_MEAN(i,k,j) + fwx_adv(i,k,j) )/count_p1*fwx_dfac
              FWY_ADV_MEAN(i,k,j)  = (local_count*FWY_ADV_MEAN(i,k,j) + fwy_adv(i,k,j) )/count_p1*fwy_dfac
            ENDIF
            IF (k .lt. kte) THEN
              !destagger in z direction for cartesian corrections
              rzxu_s = 0.5*(rzxu(i,k+1,j)+rzxu(i,k,j))
              rzyv_s = 0.5*(rzyv(i,k+1,j)+rzyv(i,k,j))
              rdzdt_s = 0.5*(rdzdt(i,k+1,j)+rdzdt(i,k,j))
              rw_diag_s = 0.5*(rw_diag(i,k+1,j) + rw_diag(i,k,j))
              WZ_MEAN(i,k,j) = (local_count*WZ_MEAN(i,k,j) + wz*rhom  )/count_p1/rho_mean
              FWZ_ADV_MEAN(i,k,j)  = (local_count*FWZ_ADV_MEAN(i,k,j) + rw_diag_s*wz )/count_p1/rho_mean_flx
              FWX_CORR_MEAN(i,k,j)  = (local_count*FWX_CORR_MEAN(i,k,j) + rzxu_s*wz )/count_p1
              FWY_CORR_MEAN(i,k,j)  = (local_count*FWY_CORR_MEAN(i,k,j) + rzyv_s*wz )/count_p1
              CORR_DWDT_MEAN(i,k,j)  = (local_count*CORR_DWDT_MEAN(i,k,j) + rdzdt_s*wz )/count_p1
            ENDIF

            IF (config_flags%output_w_fluxes_add == 1) THEN
              !2nd order fluxes
              IF (k .gt. kts) THEN
                wx2 = 0.5*(w_save(i-1,k,j) + w_save(i,k,j))
                wy2 = 0.5*(w_save(i,k,j-1) + w_save(i,k,j))
                FWX_ADV_MEAN_2ND(i,k,j) = (local_count*FWX_ADV_MEAN_2ND(i,k,j) + ru_d*wx2 )/count_p1*fwx_dfac
                FWY_ADV_MEAN_2ND(i,k,j) = (local_count*FWY_ADV_MEAN_2ND(i,k,j) + rv_d*wy2 )/count_p1*fwy_dfac
              ENDIF
              IF (k .lt. kte) THEN
                IF (k .lt. kte) THEN
                  wz2 = 0.5*(w_save(i,k+1,j) + w_save(i,k,j))
                ELSE
                  wz2 = wz
                ENDIF
                ru8z = rhom*0.5*(u_save(i,k,j) + u_save(i+1,k,j))
                rv8z = rhom*0.5*(v_save(i,k,j) + v_save(i,k,j+1))
                FWX_CORR_MEAN_DZOUT(i,k,j) = (local_count*FWX_CORR_MEAN_DZOUT(i,k,j) + ru8z*wz )/count_p1
                FWY_CORR_MEAN_DZOUT(i,k,j) = (local_count*FWY_CORR_MEAN_DZOUT(i,k,j) + rv8z*wz )/count_p1
                FWZ_ADV_MEAN_2ND(i,k,j) = (local_count*FWZ_ADV_MEAN_2ND(i,k,j) + rw_diag_s*wz2 )/count_p1/rho_mean_flx
                FWX_CORR_MEAN_2ND(i,k,j) = (local_count*FWX_CORR_MEAN_2ND(i,k,j) + rzxu_s*wz2 )/count_p1
                FWY_CORR_MEAN_2ND(i,k,j) = (local_count*FWY_CORR_MEAN_2ND(i,k,j) + rzyv_s*wz2 )/count_p1
                CORR_DWDT_MEAN_2ND(i,k,j) = (local_count*CORR_DWDT_MEAN_2ND(i,k,j) + rdzdt_s*wz2 )/count_p1
              ENDIF
            ENDIF
          ENDIF

          !average wind speed
          U_MEAN(i,k,j) = (local_count*U_MEAN(i,k,j) + u_2(i,k,j)*rhox  )/count_p1/rhox_mean
          V_MEAN(i,k,j) = (local_count*V_MEAN(i,k,j) + v_2(i,k,j)*rhoy  )/count_p1/rhoy_mean
          ZWIND_MEAN(i,k,j) = (local_count*ZWIND_MEAN(i,k,j) + w_2(i,k,j)*rhoz  )/count_p1/rhoz_mean
          WW_MEAN(i,k,j) = (local_count*WW_MEAN(i,k,j) + ww(i,k,j)*msfty(i,j))/count_p1
          WD_MEAN(i,k,j) = (local_count*WD_MEAN(i,k,j) + w_diag(i,k,j)*rhoz )/count_p1/rhoz_mean
          DPH_X_MEAN(i,k,j) = (local_count*DPH_X_MEAN(i,k,j) + dph_x(i,k,j)*rhoz )/count_p1/rhoz_mean
          DPH_Y_MEAN(i,k,j) = (local_count*DPH_Y_MEAN(i,k,j) + dph_y(i,k,j)*rhoz )/count_p1/rhoz_mean
        ENDIF

        IF (config_flags%output_sgs_fluxes) THEN
          IF (config_flags%output_tsgs_fluxes) THEN
            FTX_SGS_MEAN(i,k,j) = (local_count*FTX_SGS_MEAN(i,k,j) + ftx_sgs(i,k,j) ) / count_p1 / rhox_mean_flx
            FTY_SGS_MEAN(i,k,j) = (local_count*FTY_SGS_MEAN(i,k,j) + fty_sgs(i,k,j) ) / count_p1 / rhoy_mean_flx
            FTZ_SGS_MEAN(i,k,j) = (local_count*FTZ_SGS_MEAN(i,k,j) + ftz_sgs(i,k,j) ) / count_p1 / rhoz_mean_flx
          ENDIF
          IF (config_flags%output_qsgs_fluxes) THEN
            FQX_SGS_MEAN(i,k,j) = (local_count*FQX_SGS_MEAN(i,k,j) + fqx_sgs(i,k,j) ) / count_p1 / rhox_mean_flx
            FQY_SGS_MEAN(i,k,j) = (local_count*FQY_SGS_MEAN(i,k,j) + fqy_sgs(i,k,j) ) / count_p1 / rhoy_mean_flx
            FQZ_SGS_MEAN(i,k,j) = (local_count*FQZ_SGS_MEAN(i,k,j) + fqz_sgs(i,k,j) ) / count_p1 / rhoz_mean_flx
          ENDIF
          IF (config_flags%output_usgs_fluxes) THEN
            FUX_SGS_MEAN(i,k,j)  = (local_count*FUX_SGS_MEAN(i,k,j)  + fux_sgs(i,k,j) )  / count_p1 / rho_mean_flx
            FUZ_SGS_MEAN(i,k,j)  = (local_count*FUZ_SGS_MEAN(i,k,j)  + fuz_sgs(i,k,j) )  / count_p1 / rhoxz_mean_flx
          ENDIF
          IF (config_flags%output_vsgs_fluxes) THEN
            FVY_SGS_MEAN(i,k,j)  = (local_count*FVY_SGS_MEAN(i,k,j)  + fvy_sgs(i,k,j) )  / count_p1 / rho_mean_flx
            FVZ_SGS_MEAN(i,k,j)  = (local_count*FVZ_SGS_MEAN(i,k,j)  + fvz_sgs(i,k,j) )  / count_p1 / rhoyz_mean_flx
          ENDIF
          IF (config_flags%output_usgs_fluxes .or. config_flags%output_vsgs_fluxes) THEN
            FVX_SGS_MEAN(i,k,j)  = (local_count*FVX_SGS_MEAN(i,k,j)  + fvx_sgs(i,k,j) )  / count_p1 / rhoxy_mean_flx
          ENDIF
          IF (config_flags%output_wsgs_fluxes) THEN
            FWX_SGS_MEAN(i,k,j)  = (local_count*FWX_SGS_MEAN(i,k,j)  + fwx_sgs(i,k,j) )  / count_p1 / rhoxz_mean_flx
            FWY_SGS_MEAN(i,k,j)  = (local_count*FWY_SGS_MEAN(i,k,j)  + fwy_sgs(i,k,j) )  / count_p1 / rhoyz_mean_flx
            FWZ_SGS_MEAN(i,k,j)  = (local_count*FWZ_SGS_MEAN(i,k,j)  + fwz_sgs(i,k,j) )  / count_p1 / rho_mean_flx
          ENDIF
        ENDIF

      end DO
      end DO
      end DO

      !average additional tendencies
      IF (config_flags%output_t_fluxes .eq. 1) THEN
        DO j=jts,min(jte, jde-1)
        DO k=kts,min(kte, kde-1)
        DO i=its,min(ite, ide-1)
          T_TEND_RADLW_MEAN(i,k,j)  = (local_count*T_TEND_RADLW_MEAN(i,k,j) + t_tend_radlw(i,k,j) )/count_p1
          T_TEND_RADSW_MEAN(i,k,j)  = (local_count*T_TEND_RADSW_MEAN(i,k,j) + t_tend_radsw(i,k,j) )/count_p1
          T_TEND_MP_MEAN(i,k,j)  = (local_count*T_TEND_MP_MEAN(i,k,j) + t_tend_mp(i,k,j) )/count_p1
        ENDDO
        ENDDO
        ENDDO
      ENDIF
      IF (config_flags%output_q_fluxes .eq. 1) THEN
        DO j=jts,min(jte, jde-1)
        DO k=kts,min(kte, kde-1)
        DO i=its,min(ite, ide-1)
          Q_TEND_MP_MEAN(i,k,j)  = (local_count*Q_TEND_MP_MEAN(i,k,j) + qv_tend_mp(i,k,j) )/count_p1
        ENDDO
        ENDDO
        ENDDO
      ENDIF
      IF (config_flags%output_u_fluxes .eq. 1) THEN
        DO j=jts,min(jte, jde-1)
        DO k=kts,min(kte, kde-1)
        DO i=its,ite
          U_TEND_PG_MEAN(i,k,j)  = (local_count*U_TEND_PG_MEAN(i,k,j) + u_tend_pg(i,k,j) )/count_p1
          U_TEND_COR_CURV_MEAN(i,k,j)  = (local_count*U_TEND_COR_CURV_MEAN(i,k,j) + u_tend_cor_curv(i,k,j) )/count_p1
        ENDDO
        ENDDO
        ENDDO
      ENDIF
      IF (config_flags%output_v_fluxes .eq. 1) THEN
        DO j=jts,jte
        DO k=kts,min(kte, kde-1)
        DO i=its,min(ite, ide-1)
          V_TEND_PG_MEAN(i,k,j)  = (local_count*V_TEND_PG_MEAN(i,k,j) + v_tend_pg(i,k,j) )/count_p1
          V_TEND_COR_CURV_MEAN(i,k,j)  = (local_count*V_TEND_COR_CURV_MEAN(i,k,j) + v_tend_cor_curv(i,k,j) )/count_p1
        ENDDO
        ENDDO
        ENDDO
      ENDIF
      IF (config_flags%output_w_fluxes .eq. 1) THEN
        DO j=jts,min(jte, jde-1)
        DO k=kts,kte
        DO i=its,min(ite, ide-1)
          W_TEND_PG_MEAN(i,k,j)  = (local_count*W_TEND_PG_MEAN(i,k,j) + w_tend_pg(i,k,j) )/count_p1
          W_TEND_COR_CURV_MEAN(i,k,j)  = (local_count*W_TEND_COR_CURV_MEAN(i,k,j) + w_tend_cor_curv(i,k,j) )/count_p1
        ENDDO
        ENDDO
        ENDDO
      ENDIF

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

  subroutine get_staggered_from_fluxes(varx, vary, varz,                    &
                                       ruvar_tot, rvvar_tot, wwvar_tot,     &
                                       ru, rv, ww, var,                     &
                                       cf1, cf2, cf3, cfn, cfn1, fnm, fnp,  &
                                       x_stag, y_stag, do_w,                &
                                       i, j, k,                             &
                                       ids, ide, jds, jde, kds, kde,        &
                                       ims, ime, jms, jme, kms, kme         )

    IMPLICIT NONE

    REAL, INTENT(OUT) :: varx, vary, varz
    REAL, INTENT(IN) :: ru, rv, ww, cf1, cf2, cf3, cfn, cfn1
    LOGICAL, INTENT(IN) :: x_stag, y_stag, do_w
    INTEGER, INTENT(IN) ::  i, j, k,                       &
                            ids, ide, jds, jde, kds, kde,  &
                            ims, ime, jms, jme, kms, kme
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
                        ruvar_tot, rvvar_tot, wwvar_tot, var
    REAL, DIMENSION(kms:kme) , INTENT(IN) :: fnm, fnp


    if ((k .lt. kde) .or. (do_w)) then
      if (ru .eq. 0) then
        !cannot reconstruct correctly staggered variable -> use 2nd order = (de)staggering
        if (x_stag) then
          varx = 0.5*(var(i+1,k,j) + var(i,k,j))
        else
          varx = 0.5*(var(i-1,k,j) + var(i,k,j))
        endif
      else
        varx = ruvar_tot(i,k,j)/ru
      endif

      if (rv .eq. 0) then
        !cannot reconstruct correctly staggered variable -> use 2nd order = (de)staggering
        if (y_stag) then
          vary = 0.5*(var(i,k,j+1) + var(i,k,j))
        else
          vary = 0.5*(var(i,k,j-1) + var(i,k,j))
        endif
      else
        vary = rvvar_tot(i,k,j)/rv
      endif
    endif

    if (ww .ne. 0) then
      varz = wwvar_tot(i,k,j)/ww

    !cannot reconstruct correctly staggered variable -> use 2nd order = (de)staggering
    elseif (.not. do_w) then
      call stagger_z(varz, var(i,:,j),                   &
                     cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                     k, kms, kme, kde                    )
    else
      varz = 0.5*(var(i,k+1,j) + var(i,k,j))
    endif

  end subroutine get_staggered_from_fluxes

end MODULE module_avgflx_em
