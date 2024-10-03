from pydantic import BaseModel


class SoftwareTasks(BaseModel):
    ## Backend tasks

    database_task: int
    security_task: int
    validation_task: int
    dev_ops_task: int
    server_management_task: int
    api_setup_task: int
    api_integration_task: int
    data_backup_task: int
    backend_testing_task: int
    data_structure_task: int
    machine_learning_task: int
    scalability_task: int
    optimization_task: int
    cloud_task: int

    ## Frontend tasks
    styling_task: int
    ui_ux_task: int
    frontend_testing_task: int
    api_logic_task: int
    form_setup_task: int
    table_setup_task: int
    layout_setup_task: int
    data_display_task: int
    data_visualization_task: int
    access_control_task: int
    seo_task: int
    widget_setup_task: int
    ci_cd_task: int
    deployment_task: int
    cms_integration_task: int
    last_updated: str
    submitted_by: str
