terraform {
  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
    }
  }
}
provider "azurerm" {
  features {}
  subscription_id = "7c41a67f-73e1-4253-bf72-1f1e88c0691a"
  tenant_id       = "e66ac6ac-d605-459a-905e-fa48c629af1b"
}

resource "azurerm_resource_group" "LFS"{
	name = "LFS"
	location = "Central India"
}
