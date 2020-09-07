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
       & its, ite, jts, jte, kts, kte, config_flags,  &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1, &
       & th_mean,thx_mean,thy_mean,thz_mean,thth_mean,ruth_tot_mean,rvth_tot_mean,wwth_tot_mean, &
       & q_mean,qx_mean,qy_mean,qz_mean,qq_mean,ruq_tot_mean,rvq_tot_mean,wwq_tot_mean,          &
       & u_mean,ux_mean,uy_mean,uz_mean,ruu_tot_mean,rvu_tot_mean,wwu_tot_mean,                  &
       & v_mean,vx_mean,vy_mean,vz_mean,ruv_tot_mean,rvv_tot_mean,wwv_tot_mean,                  &
       & om_mean,wd_mean,zwind_mean,wx_mean,wy_mean,wz_mean,ruw_tot_mean,rvw_tot_mean,www_tot_mean,  &
       & corr_uth,corr_vth,corr_dthdt,                                                           &
       & corr_uq,corr_vq,corr_dqdt,                                                              &
       & corr_uu,corr_vu,corr_dudt,                                                              &
       & corr_uv,corr_vv,corr_dvdt,                                                              &
       & corr_uw,corr_vw,corr_dwdt,                                                              &
       & sgs_UTH_mean,sgs_VTH_mean,sgs_WTH_mean,                                                 &
       & sgs_UQ_mean,sgs_VQ_mean,sgs_WQ_mean,                                                    &
       & sgs_UU_mean,sgs_WU_mean,                                                                &
       & sgs_UV_mean,sgs_VV_mean,sgs_WV_mean,                                                    &
       & sgs_UW_mean,sgs_VW_mean,sgs_WW_mean,                                                    &
       & t_tend_radlw_mean,t_tend_radsw_mean,t_tend_mp_mean,q_tend_mp_mean,                      &
       & rhod_mean,z_mean                                                                        )

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte
    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags

    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(OUT) ::              &
       th_mean,thx_mean,thy_mean,thz_mean,thth_mean,ruth_tot_mean,rvth_tot_mean,wwth_tot_mean, &
       q_mean,qx_mean,qy_mean,qz_mean,qq_mean,ruq_tot_mean,rvq_tot_mean,wwq_tot_mean,          &
       u_mean,ux_mean,uy_mean,uz_mean,ruu_tot_mean,rvu_tot_mean,wwu_tot_mean,                  &
       v_mean,vx_mean,vy_mean,vz_mean,ruv_tot_mean,rvv_tot_mean,wwv_tot_mean,                  &
       om_mean,wd_mean,zwind_mean,wx_mean,wy_mean,wz_mean,ruw_tot_mean,rvw_tot_mean,www_tot_mean,  &
       corr_uth,corr_vth,corr_dthdt,                                                           &
       corr_uq,corr_vq,corr_dqdt,                                                              &
       corr_uu,corr_vu,corr_dudt,                                                              &
       corr_uv,corr_vv,corr_dvdt,                                                              &
       corr_uw,corr_vw,corr_dwdt,                                                              &
       sgs_UTH_mean,sgs_VTH_mean,sgs_WTH_mean,                                                 &
       sgs_UQ_mean,sgs_VQ_mean,sgs_WQ_mean,                                                    &
       sgs_UU_mean,sgs_WU_mean,                                                                &
       sgs_UV_mean,sgs_VV_mean,sgs_WV_mean,                                                    &
       sgs_UW_mean,sgs_VW_mean,sgs_WW_mean,                                                    &
       t_tend_radlw_mean,t_tend_radsw_mean,t_tend_mp_mean,q_tend_mp_mean,                      &
       rhod_mean,z_mean



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

    if (config_flags%output_avgfluxes .eq. 1) then
      DO j=jts,jte
         DO k=kts,kte
            DO i=its,ite
                th_mean(i,k,j) = 0.
                thx_mean(i,k,j) = 0.
                thy_mean(i,k,j) = 0.
                thz_mean(i,k,j) = 0.
                thth_mean(i,k,j) = 0.
                ruth_tot_mean(i,k,j) = 0.
                rvth_tot_mean(i,k,j) = 0.
                wwth_tot_mean(i,k,j) = 0.
                q_mean(i,k,j) = 0.
                qx_mean(i,k,j) = 0.
                qy_mean(i,k,j) = 0.
                qz_mean(i,k,j) = 0.
                qq_mean(i,k,j) = 0.
                ruq_tot_mean(i,k,j) = 0.
                rvq_tot_mean(i,k,j) = 0.
                wwq_tot_mean(i,k,j) = 0.
                u_mean(i,k,j) = 0.
                ux_mean(i,k,j) = 0.
                uy_mean(i,k,j) = 0.
                uz_mean(i,k,j) = 0.
                ruu_tot_mean(i,k,j) = 0.
                rvu_tot_mean(i,k,j) = 0.
                wwu_tot_mean(i,k,j) = 0.
                v_mean(i,k,j) = 0.
                vx_mean(i,k,j) = 0.
                vy_mean(i,k,j) = 0.
                vz_mean(i,k,j) = 0.
                ruv_tot_mean(i,k,j) = 0.
                rvv_tot_mean(i,k,j) = 0.
                wwv_tot_mean(i,k,j) = 0.
                om_mean(i,k,j) = 0.
                wd_mean(i,k,j) = 0.
                zwind_mean(i,k,j) = 0.
                wx_mean(i,k,j) = 0.
                wy_mean(i,k,j) = 0.
                wz_mean(i,k,j) = 0.
                ruw_tot_mean(i,k,j) = 0.
                rvw_tot_mean(i,k,j) = 0.
                www_tot_mean(i,k,j) = 0.
                corr_uth(i,k,j) = 0.
                corr_vth(i,k,j) = 0.
                corr_dthdt(i,k,j) = 0.
                corr_uq(i,k,j) = 0.
                corr_vq(i,k,j) = 0.
                corr_dqdt(i,k,j) = 0.
                corr_uu(i,k,j) = 0.
                corr_vu(i,k,j) = 0.
                corr_dudt(i,k,j) = 0.
                corr_uv(i,k,j) = 0.
                corr_vv(i,k,j) = 0.
                corr_dvdt(i,k,j) = 0.
                corr_uw(i,k,j) = 0.
                corr_vw(i,k,j) = 0.
                corr_dwdt(i,k,j) = 0.
                sgs_UTH_mean(i,k,j) = 0.
                sgs_VTH_mean(i,k,j) = 0.
                sgs_WTH_mean(i,k,j) = 0.
                sgs_UQ_mean(i,k,j) = 0.
                sgs_VQ_mean(i,k,j) = 0.
                sgs_WQ_mean(i,k,j) = 0.
                sgs_UU_mean(i,k,j) = 0.
                sgs_WU_mean(i,k,j) = 0.
                sgs_UV_mean(i,k,j) = 0.
                sgs_VV_mean(i,k,j) = 0.
                sgs_WV_mean(i,k,j) = 0.
                sgs_UW_mean(i,k,j) = 0.
                sgs_VW_mean(i,k,j) = 0.
                sgs_WW_mean(i,k,j) = 0.
                t_tend_radlw_mean(i,k,j) = 0.
                t_tend_radsw_mean(i,k,j) = 0.
                t_tend_mp_mean(i,k,j) = 0.
                q_tend_mp_mean(i,k,j) = 0.
                rhod_mean(i,k,j) = 0.
                z_mean(i,k,j) = 0.

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
       & TH_MEAN,THX_MEAN,THY_MEAN,THZ_MEAN,THTH_MEAN,RUTH_TOT_MEAN,RVTH_TOT_MEAN,WWTH_TOT_MEAN, &
       & Q_MEAN,QX_MEAN,QY_MEAN,QZ_MEAN,QQ_MEAN,RUQ_TOT_MEAN,RVQ_TOT_MEAN,WWQ_TOT_MEAN,          &
       & U_MEAN,UX_MEAN,UY_MEAN,UZ_MEAN,RUU_TOT_MEAN,RVU_TOT_MEAN,WWU_TOT_MEAN,                  &
       & V_MEAN,VX_MEAN,VY_MEAN,VZ_MEAN,RUV_TOT_MEAN,RVV_TOT_MEAN,WWV_TOT_MEAN,                  &
       & OM_MEAN,WD_MEAN,ZWIND_MEAN,WX_MEAN,WY_MEAN,WZ_MEAN,RUW_TOT_MEAN,RVW_TOT_MEAN,WWW_TOT_MEAN,  &
       & CORR_UTH_MEAN,CORR_VTH_MEAN,CORR_DTHDT_MEAN,                                            &
       & CORR_UQ_MEAN,CORR_VQ_MEAN,CORR_DQDT_MEAN,                                               &
       & CORR_UU_MEAN,CORR_VU_MEAN,CORR_DUDT_MEAN,                                               &
       & CORR_UV_MEAN,CORR_VV_MEAN,CORR_DVDT_MEAN,                                               &
       & CORR_UW_MEAN,CORR_VW_MEAN,CORR_DWDT_MEAN,                                               &
       & SGS_UTH_MEAN,SGS_VTH_MEAN,SGS_WTH_MEAN,                                                 &
       & SGS_UQ_MEAN,SGS_VQ_MEAN,SGS_WQ_MEAN,                                                    &
       & SGS_UU_MEAN,SGS_WU_MEAN,                                                                &
       & SGS_UV_MEAN,SGS_VV_MEAN,SGS_WV_MEAN,                                                    &
       & SGS_UW_MEAN,SGS_VW_MEAN,SGS_WW_MEAN,                                                    &
       & T_TEND_RADLW_MEAN,T_TEND_RADSW_MEAN,T_TEND_MP_MEAN,Q_TEND_MP_MEAN,                      &
       & RHOD_MEAN,Z_MEAN,                                                                       &
       & ruth_tot, rvth_tot, wwth_tot, ruq_tot, rvq_tot, wwq_tot,                                &
       & ruu_tot, rvu_tot, wwu_tot, ruv_tot, rvv_tot, wwv_tot, ruw_tot, rvw_tot, www_tot,        &
       & SGS_UTH, SGS_VTH, SGS_WTH, SGS_UQ, SGS_VQ, SGS_WQ,                                      &
       & SGS_UU, SGS_WU, SGS_UV, SGS_VV, SGS_WV, SGS_UW, SGS_VW, SGS_WW,                         &
       & t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp,                                      &
       & U, V, W, ru, rv, ww, thm, Q,                                                            &
       & ph_1, ph_2, phb, zx, zy,                                                                &
       & dnw, fnm, fnp, mut, c1h, c2h, c1f, c2f, cf1, cf2, cf3, cfn, cfn1,                       &
       & last_timestep, dt                                                                       )
    

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte

    INTEGER , INTENT(IN)        :: avgflx_count
    LOGICAL, INTENT(IN) :: do_cu, last_timestep
    REAL, INTENT(IN) :: dt, cf1, cf2, cf3, cfn, cfn1

    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
         ru_m, rv_m, ww_m,             & 
         U, V, W, ru, rv, ww,  &
         thm, Q, ph_1, ph_2, phb, zx, zy,  &
         ruth_tot, rvth_tot, wwth_tot,     &
         ruq_tot, rvq_tot, wwq_tot,        &
         ruu_tot, rvu_tot, wwu_tot,        &
         ruv_tot, rvv_tot, wwv_tot,        &
         ruw_tot, rvw_tot, www_tot,        &
         t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp, &
         SGS_UTH, SGS_VTH, SGS_WTH, SGS_UQ, SGS_VQ, SGS_WQ, &
         SGS_UU, SGS_WU, SGS_UV, SGS_VV, SGS_WV, SGS_UW, SGS_VW, SGS_WW

    REAL , DIMENSION( kms:kme ) ,    INTENT(IN   ) :: dnw, fnm, fnp, c1h, c2h, c1f, c2f
    REAL , DIMENSION( ims:ime, jms:jme ), INTENT(IN   ) :: mut



    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags


    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(IN) ::    &
         cfu1,cfd1,dfu1,efu1,dfd1,efd1
    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::            &
       TH_MEAN,THX_MEAN,THY_MEAN,THZ_MEAN,THTH_MEAN,RUTH_TOT_MEAN,RVTH_TOT_MEAN,WWTH_TOT_MEAN, &
       Q_MEAN,QX_MEAN,QY_MEAN,QZ_MEAN,QQ_MEAN,RUQ_TOT_MEAN,RVQ_TOT_MEAN,WWQ_TOT_MEAN,          &
       U_MEAN,UX_MEAN,UY_MEAN,UZ_MEAN,RUU_TOT_MEAN,RVU_TOT_MEAN,WWU_TOT_MEAN,                  &
       V_MEAN,VX_MEAN,VY_MEAN,VZ_MEAN,RUV_TOT_MEAN,RVV_TOT_MEAN,WWV_TOT_MEAN,                  &
       OM_MEAN,WD_MEAN,ZWIND_MEAN,WX_MEAN,WY_MEAN,WZ_MEAN,RUW_TOT_MEAN,RVW_TOT_MEAN,WWW_TOT_MEAN,  &
       CORR_UTH_MEAN,CORR_VTH_MEAN,CORR_DTHDT_MEAN,                                                           &
       CORR_UQ_MEAN,CORR_VQ_MEAN,CORR_DQDT_MEAN,                                                              &
       CORR_UU_MEAN,CORR_VU_MEAN,CORR_DUDT_MEAN,                                                              &
       CORR_UV_MEAN,CORR_VV_MEAN,CORR_DVDT_MEAN,                                                              &
       CORR_UW_MEAN,CORR_VW_MEAN,CORR_DWDT_MEAN,                                                              &
       SGS_UTH_MEAN,SGS_VTH_MEAN,SGS_WTH_MEAN,                                                 &
       SGS_UQ_MEAN,SGS_VQ_MEAN,SGS_WQ_MEAN,                                                    &
       SGS_UU_MEAN,SGS_WU_MEAN,                                                                &
       SGS_UV_MEAN,SGS_VV_MEAN,SGS_WV_MEAN,                                                    &
       SGS_UW_MEAN,SGS_VW_MEAN,SGS_WW_MEAN,                                                    &
       T_TEND_RADLW_MEAN,T_TEND_RADSW_MEAN,T_TEND_MP_MEAN,Q_TEND_MP_MEAN,                      &
       RHOD_MEAN,Z_MEAN

    REAL,    DIMENSION( its-1:ite , kts:kte , jts-1:jte ) ::    rho_d, th, thd

    INTEGER :: i,j,k
    REAL :: local_count, rho_8w, rhox, rhoy, rhoz, rho_mean, rhox_mean, rhoy_mean, rhoz_mean, &
            thetax, thetay, thetaz, qx, qy, qz, ux, uy, uz, vx, vy, vz, wx, wy, wz, qfac,     &
            zx_8w, zy_8w, u_8w, v_8w, dzdt, &
            corr_uth, corr_vth, corr_dthdt, &
            corr_uq, corr_vq, corr_dqdt,    &
            corr_uu, corr_vu, corr_dudt, &
            corr_uv, corr_vv, corr_dvdt, &
            corr_uw, corr_vw, corr_dwdt, &
            wdiag, muz, mu, dphidn

    local_count = real(avgflx_count)

    if (config_flags%output_avgfluxes .eq. 1) then

    DO j=jts-1,jte
    DO k=kts,kte
    DO i=its-1,ite
      !calculate dry density
      mu = c1h(k)*mut(i,j)+c2h(k)
      dphidn = ((phb(i,k+1,j)+ph_2(i,k+1,j)) - (phb(i,k,j)+ph_2(i,k,j)))/dnw(k)
      rho_d(i,k,j) = mu/(-dphidn)
      RHOD_MEAN(i,k,j) = (local_count*RHOD_MEAN(i,k,j)  + rho_d(i,k,j)  )/(local_count+1.)
      IF ( config_flags%use_theta_m .EQ. 1 ) THEN
        thd(i,k,j) = (thm(i,k,j) + t0)/(1 + rvovrd*Q(i,k,j)) - t0
        IF ( config_flags%output_dry_theta_fluxes .eq. 1) THEN
          th(i,k,j) = thd(i,k,j)
        ELSE
          th(i,k,j) = thm(i,k,j)
        ENDIF
      ELSE
        thd(i,k,j) = thm(i,k,j)
        th(i,k,j) = thm(i,k,j)
      ENDIF
    ENDDO
    ENDDO
    ENDDO

    DO j=jts,jte
      DO k=kts,kde
        DO i=its,ite

          Z_MEAN(i,k,j) = (local_count*Z_MEAN(i,k,j)  + (phb(i,k,j)+ph_2(i,k,j))/g  )/(local_count+1.)

          IF (config_flags%output_res_fluxes==1) THEN
            !stagger density
            call stagger_z(rho_8w, rho_d(i,:,j),               &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )

            if (config_flags%hesselberg_avg==1) then
              rhox = 0.5*(rho_d(i,k,j)+rho_d(i-1,k,j))
              rhoy = 0.5*(rho_d(i,k,j)+rho_d(i,k,j-1))
              rhoz = rho_8w
            else
              rhox = 1
              rhoy = 1
              rhoz = 1
            endif


            !determine staggered variables from total fluxes

            call get_staggered_from_fluxes(thetax, thetay, thetaz,      &
                                           ruth_tot, rvth_tot, wwth_tot,&
                                           ru, rv, ww, th,              &
                                           cf1, cf2, cf3, cfn, cfn1,    &
                                           i, j, k, kde,                &
                                           ims, ime, jms, jme, kms, kme  )

            !staggered q, different vertical velocity needed
            call get_staggered_from_fluxes(qx, qy, qz,                 &
                                           ruq_tot, rvq_tot, wwq_tot,  &
                                           ru_m, rv_m, ww_m, Q,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )
    !TODO: which velocity? and staggering?
            call get_staggered_from_fluxes(ux, uy, uz,                 &
                                           ruu_tot, rvu_tot, wwu_tot,  &
                                           ru_m, rv_m, ww_m, U,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )
            call get_staggered_from_fluxes(vx, vy, vz,                 &
                                           ruv_tot, rvv_tot, wwv_tot,  &
                                           ru_m, rv_m, ww_m, V,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )

           !TODO different staggering for w!!
            call get_staggered_from_fluxes(wx, wy, wz,                 &
                                           ruw_tot, rvw_tot, www_tot,  &
                                           ru_m, rv_m, ww_m, W,        &
                                           cf1, cf2, cf3, cfn, cfn1,   &
                                           i, j, k, kde,               &
                                           ims, ime, jms, jme, kms, kme  )

            !compute cartesian corrections
            zx_8w = 0.5*(zx(i,k,j)+zx(i+1,k,j))
            zy_8w = 0.5*(zy(i,k,j)+zy(i,k,j+1))
            dzdt = (ph_2(i,k,j) - ph_1(i,k,j))/(g*dt)
            call stagger_z(u_8w, U(i,:,j)+U(i+1,:,j),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )
            call stagger_z(v_8w, V(i,:,j)+V(i,:,j+1),  &
                           cf1, cf2, cf3, cfn, cfn1, fnm, fnp, &
                           i, j, k, kms, kme, kde              )

            wdiag = dzdt + zx_8w*u_8w + zy_8w*v_8w + ww(i,k,j)/(-rho_8w*g)
            muz = c1f(k)*mut(i,j)+c2f(k)

            corr_uth = rho_8w*zx_8w*u_8w*thetaz
            corr_vth = rho_8w*zy_8w*v_8w*thetaz
            corr_dthdt = rho_8w*dzdt*thetaz

            corr_uq = rho_8w*zx_8w*u_8w*qz
            corr_vq = rho_8w*zy_8w*v_8w*qz
            corr_dqdt = rho_8w*dzdt*qz

            corr_uu = rho_8w*zx_8w*u_8w*uz
            corr_vu = rho_8w*zy_8w*v_8w*uz
            corr_dudt = rho_8w*dzdt*uz

            corr_uv = rho_8w*zx_8w*u_8w*vz
            corr_vv = rho_8w*zy_8w*v_8w*vz
            corr_dvdt = rho_8w*dzdt*vz

            corr_uw = rho_8w*zx_8w*u_8w*wz
            corr_vw = rho_8w*zy_8w*v_8w*wz
            corr_dwdt = rho_8w*dzdt*wz


            if (last_timestep .and. config_flags%hesselberg_avg==1) then
              !normalize means by density
              rho_mean = RHOD_MEAN(i,k,j)
              rhox_mean = 0.5*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i-1,k,j))
              rhoy_mean = 0.5*(RHOD_MEAN(i,k,j) + RHOD_MEAN(i,k,j-1))

              call stagger_z(rhoz_mean, RHOD_MEAN(i,:,j), &
                             cf1, cf2, cf3, cfn, cfn1, fnm, fnp,      &
                             i, j, k, kms, kme, kde                   )
            ELSE
              rho_mean  = 1
              rhox_mean = 1
              rhoy_mean = 1
              rhoz_mean = 1
            ENDIF

            !mean variables
            U_MEAN(i,k,j) = (local_count*U_MEAN(i,k,j) + U(i,k,j)*rhox  )/(local_count+1.)/rhox_mean
            V_MEAN(i,k,j) = (local_count*V_MEAN(i,k,j) + V(i,k,j)*rhoy  )/(local_count+1.)/rhoy_mean
            ZWIND_MEAN(i,k,j) = (local_count*ZWIND_MEAN(i,k,j) + W(i,k,j)*rhoz  )/(local_count+1.)/rhoz_mean
            OM_MEAN(i,k,j) = (local_count*OM_MEAN(i,k,j) + ww(i,k,j)/muz*rhoz )/(local_count+1.)/rhoz_mean
            WD_MEAN(i,k,j) = (local_count*WD_MEAN(i,k,j) + wdiag*rhoz )/(local_count+1.)/rhoz_mean

            TH_MEAN(i,k,j) = (local_count*TH_MEAN(i,k,j) + thd(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            THX_MEAN(i,k,j) = (local_count*THX_MEAN(i,k,j) + thetax*rhox )/(local_count+1.)/rhox_mean
            THY_MEAN(i,k,j) = (local_count*THY_MEAN(i,k,j) + thetay*rhoy )/(local_count+1.)/rhoy_mean
            THZ_MEAN(i,k,j) = (local_count*THZ_MEAN(i,k,j) + thetaz*rhoz )/(local_count+1.)/rhoz_mean
            THTH_MEAN(i,k,j) = (local_count*THTH_MEAN(i,k,j) + thd(i,k,j)**2 )/(local_count+1.)

            Q_MEAN(i,k,j) = (local_count*Q_MEAN(i,k,j) + Q(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            QX_MEAN(i,k,j) = (local_count*QX_MEAN(i,k,j) + qx*rhox )/(local_count+1.)/rhox_mean
            QY_MEAN(i,k,j) = (local_count*QY_MEAN(i,k,j) + qy*rhoy )/(local_count+1.)/rhoy_mean
            QZ_MEAN(i,k,j) = (local_count*QZ_MEAN(i,k,j) + qz*rhoz )/(local_count+1.)/rhoz_mean
            QQ_MEAN(i,k,j) = (local_count*QQ_MEAN(i,k,j) + Q(i,k,j)*2 )/(local_count+1.)

            UX_MEAN(i,k,j) = (local_count*UX_MEAN(i,k,j) + ux*rhox )/(local_count+1.)/rhox_mean
            UY_MEAN(i,k,j) = (local_count*UY_MEAN(i,k,j) + uy*rhoy )/(local_count+1.)/rhoy_mean
            UZ_MEAN(i,k,j) = (local_count*UZ_MEAN(i,k,j) + uz*rhoz )/(local_count+1.)/rhoz_mean

            VX_MEAN(i,k,j) = (local_count*VX_MEAN(i,k,j) + vx*rhox )/(local_count+1.)/rhox_mean
            VY_MEAN(i,k,j) = (local_count*VY_MEAN(i,k,j) + vy*rhoy )/(local_count+1.)/rhoy_mean
            VZ_MEAN(i,k,j) = (local_count*VZ_MEAN(i,k,j) + vz*rhoz )/(local_count+1.)/rhoz_mean

            WX_MEAN(i,k,j) = (local_count*WX_MEAN(i,k,j) + wx*rhox )/(local_count+1.)/rhox_mean
            WY_MEAN(i,k,j) = (local_count*WY_MEAN(i,k,j) + wy*rhoy )/(local_count+1.)/rhoy_mean
            WZ_MEAN(i,k,j) = (local_count*WZ_MEAN(i,k,j) + wz*rhoz )/(local_count+1.)/rhoz_mean

            !mean total fluxes
            !TODO: momentum fluxes from advection module

            RUTH_TOT_MEAN(i,k,j)  = (local_count*RUTH_TOT_MEAN(i,k,j) + ruth_tot(i,k,j) )/(local_count+1.)
            RVTH_TOT_MEAN(i,k,j)  = (local_count*RVTH_TOT_MEAN(i,k,j) + rvth_tot(i,k,j) )/(local_count+1.)
            WWTH_TOT_MEAN(i,k,j)  = (local_count*WWTH_TOT_MEAN(i,k,j) + wwth_tot(i,k,j) )/(local_count+1.)

            RUQ_TOT_MEAN(i,k,j)  = (local_count*RUQ_TOT_MEAN(i,k,j) + ruq_tot(i,k,j) )/(local_count+1.)
            RVQ_TOT_MEAN(i,k,j)  = (local_count*RVQ_TOT_MEAN(i,k,j) + rvq_tot(i,k,j) )/(local_count+1.)
            WWQ_TOT_MEAN(i,k,j)  = (local_count*WWQ_TOT_MEAN(i,k,j) + wwq_tot(i,k,j) )/(local_count+1.)

            RUU_TOT_MEAN(i,k,j)  = (local_count*RUU_TOT_MEAN(i,k,j) + ruu_tot(i,k,j) )/(local_count+1.)
            RVU_TOT_MEAN(i,k,j)  = (local_count*RVU_TOT_MEAN(i,k,j) + rvu_tot(i,k,j) )/(local_count+1.)
            WWU_TOT_MEAN(i,k,j)  = (local_count*WWU_TOT_MEAN(i,k,j) + wwu_tot(i,k,j) )/(local_count+1.)

            RUV_TOT_MEAN(i,k,j)  = (local_count*RUV_TOT_MEAN(i,k,j) + ruv_tot(i,k,j) )/(local_count+1.)
            RVV_TOT_MEAN(i,k,j)  = (local_count*RVV_TOT_MEAN(i,k,j) + rvv_tot(i,k,j) )/(local_count+1.)
            WWV_TOT_MEAN(i,k,j)  = (local_count*WWV_TOT_MEAN(i,k,j) + wwv_tot(i,k,j) )/(local_count+1.)

            RUW_TOT_MEAN(i,k,j)  = (local_count*RUW_TOT_MEAN(i,k,j) + ruw_tot(i,k,j) )/(local_count+1.)
            RVW_TOT_MEAN(i,k,j)  = (local_count*RVW_TOT_MEAN(i,k,j) + rvw_tot(i,k,j) )/(local_count+1.)
            WWW_TOT_MEAN(i,k,j)  = (local_count*WWW_TOT_MEAN(i,k,j) + www_tot(i,k,j) )/(local_count+1.)

            !mean correction terms
            CORR_UTH_MEAN(i,k,j)  = (local_count*CORR_UTH_MEAN(i,k,j) + corr_uth )/(local_count+1.)
            CORR_VTH_MEAN(i,k,j)  = (local_count*CORR_VTH_MEAN(i,k,j) + corr_vth )/(local_count+1.)
            CORR_DTHDT_MEAN(i,k,j)  = (local_count*CORR_DTHDT_MEAN(i,k,j) + corr_dthdt )/(local_count+1.)

            CORR_UQ_MEAN(i,k,j)  = (local_count*CORR_UQ_MEAN(i,k,j) + corr_uq )/(local_count+1.)
            CORR_VQ_MEAN(i,k,j)  = (local_count*CORR_VQ_MEAN(i,k,j) + corr_vq )/(local_count+1.)
            CORR_DQDT_MEAN(i,k,j)  = (local_count*CORR_DQDT_MEAN(i,k,j) + corr_dqdt )/(local_count+1.)

            CORR_UU_MEAN(i,k,j)  = (local_count*CORR_UU_MEAN(i,k,j) + corr_uu )/(local_count+1.)
            CORR_VU_MEAN(i,k,j)  = (local_count*CORR_VU_MEAN(i,k,j) + corr_vu )/(local_count+1.)
            CORR_DUDT_MEAN(i,k,j)  = (local_count*CORR_DUDT_MEAN(i,k,j) + corr_dudt )/(local_count+1.)

            CORR_UV_MEAN(i,k,j)  = (local_count*CORR_UV_MEAN(i,k,j) + corr_uv )/(local_count+1.)
            CORR_VV_MEAN(i,k,j)  = (local_count*CORR_VV_MEAN(i,k,j) + corr_vv )/(local_count+1.)
            CORR_DVDT_MEAN(i,k,j)  = (local_count*CORR_DVDT_MEAN(i,k,j) + corr_dvdt )/(local_count+1.)

            CORR_UW_MEAN(i,k,j)  = (local_count*CORR_UW_MEAN(i,k,j) + corr_uw )/(local_count+1.)
            CORR_VW_MEAN(i,k,j)  = (local_count*CORR_VW_MEAN(i,k,j) + corr_vw )/(local_count+1.)
            CORR_DWDT_MEAN(i,k,j)  = (local_count*CORR_DWDT_MEAN(i,k,j) + corr_dwdt )/(local_count+1.)

          ENDIF

          T_TEND_RADLW_MEAN(i,k,j)  = (local_count*T_TEND_RADLW_MEAN(i,k,j) + t_tend_radlw(i,k,j) )/(local_count+1.)
          T_TEND_RADSW_MEAN(i,k,j)  = (local_count*T_TEND_RADSW_MEAN(i,k,j) + t_tend_radsw(i,k,j) )/(local_count+1.)
          T_TEND_MP_MEAN(i,k,j)  = (local_count*T_TEND_MP_MEAN(i,k,j) + t_tend_mp(i,k,j) )/(local_count+1.)
          Q_TEND_MP_MEAN(i,k,j)  = (local_count*Q_TEND_MP_MEAN(i,k,j) + qv_tend_mp(i,k,j) )/(local_count+1.)
          !TODO: temperature tendencies from cumulus scheme; small time step tendencies from momentum
          IF (config_flags%output_sgs_fluxes==1) THEN
            sgs_UU_mean(i,k,j)  = (local_count*sgs_UU_mean(i,k,j)  + sgs_UU(i,k,j) )  / (local_count+1.)
            sgs_VV_mean(i,k,j)  = (local_count*sgs_VV_mean(i,k,j)  + sgs_VV(i,k,j)  )  / (local_count+1.)
            sgs_WW_mean(i,k,j)  = (local_count*sgs_WW_mean(i,k,j)  + sgs_WW(i,k,j)  )  / (local_count+1.)
            sgs_UV_mean(i,k,j)  = (local_count*sgs_UV_mean(i,k,j)  + sgs_UV(i,k,j) )  / (local_count+1.)
            sgs_UW_mean(i,k,j)  = (local_count*sgs_UW_mean(i,k,j)  + sgs_UW(i,k,j) )  / (local_count+1.)
            sgs_VW_mean(i,k,j)  = (local_count*sgs_VW_mean(i,k,j)  + sgs_VW(i,k,j) )  / (local_count+1.)
            sgs_WU_mean(i,k,j)  = (local_count*sgs_WU_mean(i,k,j)  + sgs_WU(i,k,j) )  / (local_count+1.)
            sgs_WV_mean(i,k,j)  = (local_count*sgs_WV_mean(i,k,j)  + sgs_WV(i,k,j) )  / (local_count+1.)

            sgs_UTH_mean(i,k,j) = (local_count*sgs_UTH_mean(i,k,j) + sgs_UTH(i,k,j) ) / (local_count+1.)
            sgs_VTH_mean(i,k,j) = (local_count*sgs_VTH_mean(i,k,j) + sgs_VTH(i,k,j) ) / (local_count+1.)
            sgs_WTH_mean(i,k,j) = (local_count*sgs_WTH_mean(i,k,j) + sgs_WTH(i,k,j) ) / (local_count+1.)

            sgs_UQ_mean(i,k,j) = (local_count*sgs_UQ_mean(i,k,j) + sgs_UQ(i,k,j) ) / (local_count+1.)
            sgs_VQ_mean(i,k,j) = (local_count*sgs_VQ_mean(i,k,j) + sgs_VQ(i,k,j) ) / (local_count+1.)
            sgs_WQ_mean(i,k,j) = (local_count*sgs_WQ_mean(i,k,j) + sgs_WQ(i,k,j) ) / (local_count+1.)
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
                       i, j, k, kms, kme, kde              )

    REAL, INTENT(OUT) :: var_stag
    REAL, INTENT(IN) :: cf1, cf2, cf3, cfn, cfn1
    INTEGER, INTENT(IN) ::  i, j, k, kms, kme, kde
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
                                       cf1, cf2, cf3, cfn, cfn1,        &
                                       i, j, k, kde,                    &
                                       ims, ime, jms, jme, kms, kme     )

    IMPLICIT NONE

    REAL, INTENT(OUT) :: varx, vary, varz
    REAL, INTENT(IN) :: cf1, cf2, cf3, cfn, cfn1
    INTEGER, INTENT(IN) ::  i, j, k, kde,             &
                            ims, ime, jms, jme, kms, kme
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
                        ruvar_tot, rvvar_tot, wwvar_tot, &
                        ru, rv, ww, var

    varx = ruvar_tot(i,k,j)/ru(i,k,j)
    vary = rvvar_tot(i,k,j)/rv(i,k,j)
    if ( k .eq. kde) then
      varz = cfn*var(i,kde-1,j)+cfn1*var(i,kde-2,j)
    elseif ( k .eq. 1) then
      varz = cf1*var(i,1,j)+cf2*var(i,2,j)+cf3*var(i,3,j)
    else
      varz = wwvar_tot(i,k,j)/ww(i,k,j)
    endif
  end subroutine get_staggered_from_fluxes

end MODULE module_avgflx_em
