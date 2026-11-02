import { EmptyState, PageHeading, Panel } from "../components";

function DevicesPage() {
  return (
    <>
      <PageHeading
        eyebrow="DEVICE MONITORING"
        title="Devices"
        description="Device inventory and device health metrics."
      />
      <Panel title="Device inventory" subtitle="Connected devices and activity">
        <EmptyState
          icon="▣"
          message="Device-level records are not included in the configured CICIDS2017 traffic data or current API."
        />
      </Panel>
    </>
  );
}

export default DevicesPage;
